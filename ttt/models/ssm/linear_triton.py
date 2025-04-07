import math
from functools import partial

import torch
from torch.distributed._tensor import Shard
from torch.distributed._tensor.experimental import local_map

from ttt.models.ssm.kernels.linear_backward import ttt_linear_scan_backward
from ttt.models.ssm.kernels.linear_forward import ttt_linear_scan_forward


class TritonLinear(torch.autograd.Function):
    sharded_mode = False

    @staticmethod
    def forward(
        ctx,
        ttt_norm_weight,
        ttt_norm_bias,
        W1_init,
        b1_init,
        XQ_batch,
        XV_batch,
        XK_batch,
        eta_batch,
        checkpoint_group_size,
    ) -> torch.Tensor:
        if TritonLinear.sharded_mode:
            return TritonLinear.forward_sharded(
                ctx,
                ttt_norm_weight,
                ttt_norm_bias,
                W1_init,
                b1_init,
                XQ_batch,
                XV_batch,
                XK_batch,
                eta_batch,
                checkpoint_group_size,
            )
        else:
            return TritonLinear.forward_unsharded(
                ctx,
                ttt_norm_weight,
                ttt_norm_bias,
                W1_init,
                b1_init,
                XQ_batch,
                XV_batch,
                XK_batch,
                eta_batch,
                checkpoint_group_size,
            )

    @staticmethod
    def backward(ctx, grad_L_XQW_batch):
        if TritonLinear.sharded_mode:
            return TritonLinear.backward_sharded(ctx, grad_L_XQW_batch)
        else:
            return TritonLinear.backward_unsharded(ctx, grad_L_XQW_batch)

    @staticmethod
    def _forward_core(
        ctx,
        ttt_norm_weight,
        ttt_norm_bias,
        W1_init,
        b1_init,
        XQ_batch,
        XV_batch,
        XK_batch,
        eta_batch,
        checkpoint_group_size,
    ):
        B, NH, NC, CS, F = XQ_batch.shape
        K = math.ceil(NC / checkpoint_group_size)

        device = XQ_batch.device
        mp_dtype = XQ_batch.dtype  # NOTE: FP32 / BF16 depending on mixed precision policy

        # Output pointers
        W1_last = torch.empty(B, NH, F, F, device=device, dtype=torch.float32)
        b1_last = torch.empty(B, NH, 1, F, device=device, dtype=torch.float32)
        XQW_batch = torch.empty(B, NH, NC, CS, F, device=device, dtype=torch.float32)

        # Context pointers
        W1_checkpoints = torch.empty(B, NH, K, F, F, device=device, dtype=torch.float32)
        b1_checkpoints = torch.empty(B, NH, K, 1, F, device=device, dtype=torch.float32)

        # Strides
        CS_F_stride = CS * F
        F_F_stride = F * F
        CS_CS_stride = CS * CS
        F_stride = F

        grid = (B, NH)

        ttt_linear_scan_forward[grid](
            # Scan inputs
            ttt_norm_weight.contiguous(),
            ttt_norm_bias.contiguous(),
            W1_init.to(torch.float32).contiguous(),
            b1_init.to(torch.float32).contiguous(),
            XQ_batch.contiguous(),
            XV_batch.contiguous(),
            XK_batch.contiguous(),
            eta_batch.contiguous(),
            # Outputs
            W1_last.contiguous(),
            b1_last.contiguous(),
            XQW_batch.contiguous(),
            # Context pointers
            W1_checkpoints.contiguous(),
            b1_checkpoints.contiguous(),
            # Strides
            CS_F_stride,
            F_F_stride,
            CS_CS_stride,
            F_stride,
            # Constant expressions
            NH,
            NC,
            CS,
            F,
            K,
            checkpoint_group_size,
        )

        # Torch context requires tensors only
        checkpoint_shapes = torch.tensor([K, checkpoint_group_size])

        ctx.save_for_backward(
            XQ_batch,
            XV_batch,
            XK_batch,
            eta_batch,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_checkpoints,
            b1_checkpoints,
            checkpoint_shapes,
        )

        return XQW_batch.to(mp_dtype)

    @staticmethod
    def _backward_core(ctx, grad_L_XQW_batch):
        (
            XQ_batch,
            XV_batch,
            XK_batch,
            eta_batch,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_checkpoints,
            b1_checkpoints,
            checkpoint_shapes,
        ) = ctx.saved_tensors

        B, NH, NC, CS, F = XQ_batch.shape
        K, checkpoint_group_size = checkpoint_shapes[0].item(), checkpoint_shapes[1].item()

        device = XQ_batch.device
        mp_dtype = XQ_batch.dtype  # NOTE: FP32 / BF16 depending on mixed precision policy
        intermediate_dtype = torch.float32

        # Intermediate buffers for each checkpoint group
        W1_init_group = torch.empty(B, NH, checkpoint_group_size, F, F, device=device, dtype=torch.float32)
        grad_L_W1_last = torch.zeros(B, NH, F, F, device=device, dtype=torch.float32)
        grad_L_b1_last = torch.zeros(B, NH, 1, F, device=device, dtype=torch.float32)

        x_hat_ln_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=intermediate_dtype)
        std_ln_group = torch.empty(B, NH, checkpoint_group_size, CS, 1, device=device, dtype=intermediate_dtype)
        grad_l_wrt_Z1_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=intermediate_dtype)
        x_hat_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, F, device=device, dtype=intermediate_dtype)
        grad_x_hat_fused_group = torch.empty(
            B, NH, checkpoint_group_size, CS, F, device=device, dtype=intermediate_dtype
        )
        grad_output_fused_group = torch.empty(
            B, NH, checkpoint_group_size, CS, F, device=device, dtype=intermediate_dtype
        )
        std_fused_group = torch.empty(B, NH, checkpoint_group_size, CS, 1, device=device, dtype=intermediate_dtype)

        # NOTE: Sum over batch post-kernel to avoid sync barrier
        grad_L_ttt_norm_weight = torch.empty(B, NH, 1, F, device=device, dtype=torch.float32)
        grad_L_ttt_norm_bias = torch.empty(B, NH, 1, F, device=device, dtype=torch.float32)

        grad_L_W1_init = torch.empty(B, NH, F, F, device=device, dtype=torch.float32)
        grad_L_b1_init = torch.empty(B, NH, 1, F, device=device, dtype=torch.float32)

        grad_L_XQ = torch.empty(B, NH, NC, CS, F, device=device, dtype=torch.float32)
        grad_L_XV = torch.empty(B, NH, NC, CS, F, device=device, dtype=torch.float32)
        grad_L_XK = torch.empty(B, NH, NC, CS, F, device=device, dtype=torch.float32)
        grad_L_eta = torch.empty(B, NH, NC, CS, CS, device=device, dtype=torch.float32)

        CS_F_stride = CS * F
        F_F_stride = F * F
        CS_CS_stride = CS * CS
        F_stride = F

        grid = (B, NH)

        ttt_linear_scan_backward[grid](
            XQ_batch.contiguous(),
            XV_batch.contiguous(),
            XK_batch.contiguous(),
            eta_batch.contiguous(),
            ttt_norm_weight.contiguous(),
            ttt_norm_bias.contiguous(),
            W1_checkpoints.contiguous(),
            b1_checkpoints.contiguous(),
            # Upstream gradients
            grad_L_W1_last.to(torch.float32).contiguous(),
            grad_L_b1_last.to(torch.float32).contiguous(),
            grad_L_XQW_batch.contiguous(),
            # Intermediate buffers,
            W1_init_group.contiguous(),
            x_hat_ln_group.contiguous(),
            std_ln_group.contiguous(),
            grad_l_wrt_Z1_group.contiguous(),
            x_hat_fused_group.contiguous(),
            grad_x_hat_fused_group.contiguous(),
            grad_output_fused_group.contiguous(),
            std_fused_group.contiguous(),
            # Output buffers
            grad_L_ttt_norm_weight.contiguous(),
            grad_L_ttt_norm_bias.contiguous(),
            grad_L_W1_init.contiguous(),
            grad_L_b1_init.contiguous(),
            grad_L_XQ.contiguous(),
            grad_L_XV.contiguous(),
            grad_L_XK.contiguous(),
            grad_L_eta.contiguous(),
            # Strides
            CS_F_stride,
            F_F_stride,
            CS_CS_stride,
            F_stride,
            # Constant expressions
            NH,
            NC,
            CS,
            F,
            K,
            checkpoint_group_size,
        )

        assert (grad_L_W1_last == 0).all(), "grad_L_W1_last is not all zero"
        assert (grad_L_b1_last == 0).all(), "grad_L_b1_last is not all zero"

        grad_L_ttt_norm_weight = grad_L_ttt_norm_weight.sum(dim=0).squeeze(1)
        grad_L_ttt_norm_bias = grad_L_ttt_norm_bias.sum(dim=0).squeeze(1)

        return (
            grad_L_ttt_norm_weight.to(mp_dtype),
            grad_L_ttt_norm_bias.to(mp_dtype),
            grad_L_W1_init.to(mp_dtype),
            grad_L_b1_init.to(mp_dtype),
            grad_L_XQ.to(mp_dtype),
            grad_L_XV.to(mp_dtype),
            grad_L_XK.to(mp_dtype),
            grad_L_eta.to(mp_dtype),
            None,
            None,
        )

    @staticmethod
    @partial(
        local_map,
        in_placements=(
            None,  # ctx
            [Shard(0)],  # ttt_norm_weight
            [Shard(0)],  # ttt_norm_bias
            [Shard(1)],  # W1_init
            [Shard(1)],  # b1_init
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
        XQ_batch,
        XV_batch,
        XK_batch,
        eta_batch,
        checkpoint_group_size,
    ):
        return TritonLinear._forward_core(
            ctx,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_init,
            b1_init,
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
        XQ_batch,
        XV_batch,
        XK_batch,
        eta_batch,
        checkpoint_group_size,
    ):
        return TritonLinear._forward_core(
            ctx,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_init,
            b1_init,
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
            [Shard(1)],  # grad_L_XQ
            [Shard(1)],  # grad_L_XV
            [Shard(1)],  # grad_L_XK
            [Shard(1)],  # grad_L_eta
            None,
            None,
        ),
    )
    def backward_sharded(ctx, grad_L_XQW_batch):
        return TritonLinear._backward_core(ctx, grad_L_XQW_batch)

    @staticmethod
    @partial(local_map, in_placements=None, out_placements=None)
    def backward_unsharded(ctx, grad_L_XQW_batch):
        return TritonLinear._backward_core(ctx, grad_L_XQW_batch)
