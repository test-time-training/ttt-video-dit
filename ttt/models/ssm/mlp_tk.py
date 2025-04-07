import math
from functools import partial

import torch
from torch.distributed._tensor import Shard
from torch.distributed._tensor.experimental import local_map


class TkMLP(torch.autograd.Function):
    sharded_mode = False

    @staticmethod
    def forward(
        ctx,
        ttt_norm_weight,
        ttt_norm_bias,
        W1_init,
        b1_init,
        W2_init,
        b2_init,
        XQ_batch,
        XV_batch,
        XK_batch,
        eta_batch,
        checkpoint_group_size,
    ):
        if TkMLP.sharded_mode:
            return TkMLP.forward_sharded(
                ctx,
                ttt_norm_weight,
                ttt_norm_bias,
                W1_init,
                b1_init,
                W2_init,
                b2_init,
                XQ_batch,
                XV_batch,
                XK_batch,
                eta_batch,
                checkpoint_group_size,
            )
        else:
            return TkMLP.forward_unsharded(
                ctx,
                ttt_norm_weight,
                ttt_norm_bias,
                W1_init,
                b1_init,
                W2_init,
                b2_init,
                XQ_batch,
                XV_batch,
                XK_batch,
                eta_batch,
                checkpoint_group_size,
            )

    @staticmethod
    def backward(ctx, grad_L_XQW_batch):
        if TkMLP.sharded_mode:
            return TkMLP.backward_sharded(ctx, grad_L_XQW_batch)
        else:
            return TkMLP.backward_unsharded(ctx, grad_L_XQW_batch)

    @staticmethod
    def _forward_core(
        ctx,
        ttt_norm_weight,
        ttt_norm_bias,
        W1_init,
        b1_init,
        W2_init,
        b2_init,
        XQ_batch,
        XV_batch,
        XK_batch,
        eta_batch,
        checkpoint_group_size,
    ):
        import test_time_training as ttt_mlp

        B, NH, NC, CS, F = XQ_batch.shape
        K = math.ceil(NC / checkpoint_group_size)

        device = XQ_batch.device
        mp_dtype = XQ_batch.dtype  # NOTE: FP32 / BF16 depending on mixed precision policy
        intermediate_dtype = torch.float32

        assert mp_dtype == torch.bfloat16, "Thunderkittens kernel must run in mixed-precision bfloat16."

        # Output pointers
        XQW_batch = torch.zeros(B, NH, NC, CS, F, device=device, dtype=mp_dtype).contiguous()

        # Context pointers (Checkpoints are always saved in Float32)
        W1_checkpoints = torch.empty(B, NH, K, F, F * 4, device=device, dtype=torch.float32).contiguous()
        b1_checkpoints = torch.empty(B, NH, K, 1, F * 4, device=device, dtype=torch.float32).contiguous()
        W2_checkpoints = torch.empty(B, NH, K, F * 4, F, device=device, dtype=torch.float32).contiguous()
        b2_checkpoints = torch.empty(B, NH, K, 1, F, device=device, dtype=torch.float32).contiguous()

        # Cast and make inputs contiguous
        XQ_batch = XQ_batch.contiguous()
        XV_batch = XV_batch.contiguous()
        XK_batch = XK_batch.contiguous()
        eta_batch = eta_batch.to(mp_dtype).contiguous()
        last_eta = eta_batch[:, :, :, -1, :, None].contiguous()

        W1_init = W1_init.to(torch.float32).contiguous()
        b1_init = b1_init.to(torch.float32).contiguous()
        W2_init = W2_init.to(torch.float32).contiguous()
        b2_init = b2_init.to(torch.float32).contiguous()

        ttt_norm_weight = ttt_norm_weight.unsqueeze(0).unsqueeze(2).to(torch.float32).contiguous()
        ttt_norm_bias = ttt_norm_bias.unsqueeze(0).unsqueeze(2).to(torch.float32).contiguous()

        # Call tk kernel
        ttt_mlp.ttt_forward(
            XQ_batch,
            XK_batch,
            XV_batch,
            last_eta,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_init,
            b1_init,
            W2_init,
            b2_init,
            W1_checkpoints,
            b1_checkpoints,
            W2_checkpoints,
            b2_checkpoints,
            XQW_batch,
            checkpoint_group_size,
        )

        # Torch context requires tensors only
        checkpoint_shapes = torch.tensor([K, checkpoint_group_size])
        ctx.save_for_backward(
            XQ_batch,
            XV_batch,
            XK_batch,
            last_eta,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_checkpoints,
            b1_checkpoints,
            W2_checkpoints,
            b2_checkpoints,
            checkpoint_shapes,
            XQW_batch,
        )

        return XQW_batch.to(mp_dtype)

    @staticmethod
    def _backward_core(ctx, grad_L_XQW_batch):
        import test_time_training as ttt_mlp

        (
            XQ_batch,
            XV_batch,
            XK_batch,
            last_eta,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_checkpoints,
            b1_checkpoints,
            W2_checkpoints,
            b2_checkpoints,
            checkpoint_shapes,
            XQW_batch,
        ) = ctx.saved_tensors

        B, NH, NC, CS, F = XQ_batch.shape
        K, checkpoint_group_size = checkpoint_shapes[0].item(), checkpoint_shapes[1].item()

        device = XQ_batch.device
        mp_dtype = XQ_batch.dtype  # NOTE: FP32 / BF16 depending on mixed precision policy

        grad_L_W1_last = torch.zeros(B, NH, F, F * 4, device=device, dtype=torch.float32)
        grad_L_b1_last = torch.zeros(B, NH, 1, F * 4, device=device, dtype=torch.float32)
        grad_L_W2_last = torch.zeros(B, NH, F * 4, F, device=device, dtype=torch.float32)
        grad_L_b2_last = torch.zeros(B, NH, 1, F, device=device, dtype=torch.float32)

        # Cast upstream grads
        grad_L_W1_last = grad_L_W1_last.to(torch.float32).contiguous()
        grad_L_b1_last = grad_L_b1_last.to(torch.float32).contiguous()
        grad_L_W2_last = grad_L_W2_last.to(torch.float32).contiguous()
        grad_L_b2_last = grad_L_b2_last.to(torch.float32).contiguous()
        grad_L_XQW_batch = grad_L_XQW_batch.to(torch.bfloat16).contiguous()

        # Remat buffers
        W1_init_group = torch.empty(B, NH, checkpoint_group_size, F, F * 4, device=device, dtype=torch.float32)
        b1_init_group = torch.empty(B, NH, checkpoint_group_size, 1, F * 4, device=device, dtype=torch.float32)
        W2_init_group = torch.empty(B, NH, checkpoint_group_size, F * 4, F, device=device, dtype=torch.float32)
        b2_init_group = torch.empty(B, NH, checkpoint_group_size, 1, F, device=device, dtype=torch.float32)

        x_hat_ln_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=torch.bfloat16)
        std_ln_group = torch.empty(B, NH, checkpoint_group_size, CS, 1, device=device, dtype=torch.float32)

        X2_group = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=torch.bfloat16)
        Z1_group = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=torch.bfloat16)
        Z1_bar_group = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=torch.bfloat16)
        X2_bar_group = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=torch.bfloat16)

        grad_l_wrt_Z2_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=torch.bfloat16)
        grad_l_wrt_Z1_group = torch.empty(B, NH, checkpoint_group_size, CS, F * 4, device=device, dtype=torch.bfloat16)
        x_hat_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=torch.bfloat16)
        grad_x_hat_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=torch.bfloat16)
        grad_output_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=torch.bfloat16)
        std_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, 1, device=device, dtype=torch.float32)

        # Final gradients
        grad_L_XQ = torch.zeros(B, NH, NC, CS, F, device=device, dtype=torch.bfloat16)
        grad_L_XV = torch.zeros(B, NH, NC, CS, F, device=device, dtype=torch.bfloat16)
        grad_L_XK = torch.zeros(B, NH, NC, CS, F, device=device, dtype=torch.bfloat16)
        grad_L_last_eta = torch.zeros(B, NH, NC, CS, 1, device=device, dtype=torch.bfloat16).contiguous()

        # NOTE: Sum over batch post-kernel to avoid sync barrier
        grad_L_ttt_norm_weight = torch.zeros(B, NH, 1, F, device=device, dtype=torch.float32)
        grad_L_ttt_norm_bias = torch.zeros(B, NH, 1, F, device=device, dtype=torch.float32)

        grad_L_W1_init = torch.zeros_like(grad_L_W1_last).contiguous()
        grad_L_b1_init = torch.zeros_like(grad_L_b1_last).contiguous()
        grad_L_W2_init = torch.zeros_like(grad_L_W2_last).contiguous()
        grad_L_b2_init = torch.zeros_like(grad_L_b2_last).contiguous()

        ttt_mlp.ttt_backward(
            XQ_batch.contiguous(),
            XK_batch.contiguous(),
            XV_batch.contiguous(),
            last_eta.contiguous(),
            ttt_norm_weight.contiguous(),
            ttt_norm_bias.contiguous(),
            # Checkpoints
            W1_checkpoints.contiguous(),
            b1_checkpoints.contiguous(),
            W2_checkpoints.contiguous(),
            b2_checkpoints.contiguous(),
            XQW_batch.contiguous(),
            # Rematted Buffers
            W1_init_group.contiguous(),
            b1_init_group.contiguous(),
            W2_init_group.contiguous(),
            b2_init_group.contiguous(),
            x_hat_ln_group.contiguous(),
            std_ln_group.contiguous(),
            X2_group.contiguous(),
            Z1_group.contiguous(),
            Z1_bar_group.contiguous(),
            X2_bar_group.contiguous(),
            grad_l_wrt_Z2_group.contiguous(),
            grad_l_wrt_Z1_group.contiguous(),
            x_hat_fused_group.contiguous(),
            grad_x_hat_fused_group.contiguous(),
            grad_output_fused_group.contiguous(),
            std_fused_group.contiguous(),
            # Upstream grads
            grad_L_W1_last.contiguous(),
            grad_L_b1_last.contiguous(),
            grad_L_W2_last.contiguous(),
            grad_L_b2_last.contiguous(),
            grad_L_XQW_batch.contiguous(),
            # Output grads
            grad_L_ttt_norm_weight.contiguous(),
            grad_L_ttt_norm_bias.contiguous(),
            grad_L_W1_init.contiguous(),
            grad_L_b1_init.contiguous(),
            grad_L_W2_init.contiguous(),
            grad_L_b2_init.contiguous(),
            grad_L_last_eta.contiguous(),
            grad_L_XQ.contiguous(),
            grad_L_XK.contiguous(),
            grad_L_XV.contiguous(),
            checkpoint_group_size,
        )

        grad_L_ttt_norm_weight = grad_L_ttt_norm_weight.sum(dim=0).squeeze(1)
        grad_L_ttt_norm_bias = grad_L_ttt_norm_bias.sum(dim=0).squeeze(1)

        grad_L_eta = torch.nn.functional.pad(grad_L_last_eta.transpose(-2, -1), (0, 0, 63, 0))

        return (
            grad_L_ttt_norm_weight.to(mp_dtype),
            grad_L_ttt_norm_bias.to(mp_dtype),
            grad_L_W1_init.to(mp_dtype),
            grad_L_b1_init.to(mp_dtype),
            grad_L_W2_init.to(mp_dtype),
            grad_L_b2_init.to(mp_dtype),
            grad_L_XQ.to(mp_dtype),
            grad_L_XV.to(mp_dtype),
            grad_L_XK.to(mp_dtype),
            grad_L_eta.to(mp_dtype),
            None,
        )

    # --- Decorated wrappers for sharded vs. unsharded modes ---
    @staticmethod
    @partial(
        local_map,
        in_placements=(
            None,
            [Shard(0)],  # ttt_norm_weight
            [Shard(0)],  # ttt_norm_bias
            [Shard(1)],  # W1_init
            [Shard(1)],  # b1_init
            [Shard(1)],  # W2_init
            [Shard(1)],  # b2_init
            [Shard(1)],  # XQ_batch
            [Shard(1)],  # XV_batch
            [Shard(1)],  # XK_batch
            [Shard(1)],  # eta_batch
            None,  # checkpoint_group_size
        ),
        out_placements=([Shard(1)],),  # XQW_batch
    )
    def forward_sharded(
        ctx,
        ttt_norm_weight,
        ttt_norm_bias,
        W1_init,
        b1_init,
        W2_init,
        b2_init,
        XQ_batch,
        XV_batch,
        XK_batch,
        eta_batch,
        checkpoint_group_size,
    ):
        return TkMLP._forward_core(
            ctx,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_init,
            b1_init,
            W2_init,
            b2_init,
            XQ_batch,
            XV_batch,
            XK_batch,
            eta_batch,
            checkpoint_group_size,
        )

    @staticmethod
    @partial(local_map, in_placements=None, out_placements=None)
    def forward_unsharded(
        ctx,
        ttt_norm_weight,
        ttt_norm_bias,
        W1_init,
        b1_init,
        W2_init,
        b2_init,
        XQ_batch,
        XV_batch,
        XK_batch,
        eta_batch,
        checkpoint_group_size,
    ):
        return TkMLP._forward_core(
            ctx,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_init,
            b1_init,
            W2_init,
            b2_init,
            XQ_batch,
            XV_batch,
            XK_batch,
            eta_batch,
            checkpoint_group_size,
        )

    @staticmethod
    @partial(
        local_map,
        in_placements=(
            None,
            [Shard(1)],  # grad_L_XQW_batch
        ),
        out_placements=(
            [Shard(0)],  # grad_L_ttt_norm_weight
            [Shard(0)],  # grad_L_ttt_norm_bias
            [Shard(1)],  # grad_L_W1_init
            [Shard(1)],  # grad_L_b1_init
            [Shard(1)],  # grad_L_W2_init
            [Shard(1)],  # grad_L_b2_init
            [Shard(1)],  # grad_L_XQ
            [Shard(1)],  # grad_L_XV
            [Shard(1)],  # grad_L_XK
            [Shard(1)],  # grad_L_eta
            None,  # checkpoint_group_size
        ),
    )
    def backward_sharded(ctx, grad_L_XQW_batch):
        return TkMLP._backward_core(ctx, grad_L_XQW_batch)

    @staticmethod
    @partial(local_map, in_placements=None, out_placements=None)
    def backward_unsharded(ctx, grad_L_XQW_batch):
        return TkMLP._backward_core(ctx, grad_L_XQW_batch)
