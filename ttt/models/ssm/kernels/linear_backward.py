import triton
import triton.language as tl


@triton.jit
def ttt_linear_mini_batch_forward(
    W1_init,
    b1_init,
    ln_weight,
    ln_bias,
    XQ_mini_batch,
    XK_mini_batch,
    XV_mini_batch,
    last_eta_mini_batch,
    CS: tl.constexpr,
    F: tl.constexpr,
    mp_dtype,
):
    # Stage 1: MatMul
    Z1 = tl.dot(XK_mini_batch.to(mp_dtype), W1_init.to(mp_dtype)) + b1_init
    reconstruction_target = XV_mini_batch - XK_mini_batch

    # Stage 2: LnFusedL2BWD
    mu_fused = (tl.sum(Z1, axis=1) / F)[:, None]
    var_fused = (tl.sum((Z1 - mu_fused) * (Z1 - mu_fused), axis=1) / F)[:, None]

    std_fused = tl.sqrt(var_fused + 1e-6)
    x_hat_fused = (Z1 - mu_fused) / std_fused

    y = ln_weight * x_hat_fused + ln_bias
    grad_output_fused = y - reconstruction_target
    grad_x_hat_fused = grad_output_fused * ln_weight

    grad_l_wrt_Z1 = (
        (1.0 / F)
        * (
            F * grad_x_hat_fused
            - tl.sum(grad_x_hat_fused, axis=1)[:, None]
            - x_hat_fused * tl.sum(grad_x_hat_fused * x_hat_fused, axis=1)[:, None]
        )
        / std_fused
    )

    W1_last = W1_init - tl.dot(tl.trans(last_eta_mini_batch * XK_mini_batch).to(mp_dtype), grad_l_wrt_Z1.to(mp_dtype))
    b1_last = b1_init - tl.sum(last_eta_mini_batch * grad_l_wrt_Z1, axis=0)[None, :]

    Z1_bar = tl.dot(XQ_mini_batch.to(mp_dtype), W1_last.to(mp_dtype)) + b1_last

    # Stage 4: LN
    mu_ln = tl.sum(Z1_bar, axis=1)[:, None] / F
    var_ln = tl.sum((Z1_bar - mu_ln) * (Z1_bar - mu_ln), axis=1)[:, None] / F
    std_ln = tl.sqrt(var_ln + 1e-6)
    x_hat_ln = (Z1_bar - mu_ln) / std_ln

    Z1_bar_ln = ln_weight * x_hat_ln + ln_bias

    XQW_mini_batch = XQ_mini_batch + Z1_bar_ln

    return (
        XQW_mini_batch,
        W1_last,
        b1_last,
        x_hat_ln,
        std_ln,
        grad_l_wrt_Z1,
        x_hat_fused,
        grad_x_hat_fused,
        grad_output_fused,
        std_fused,
    )


@triton.jit
def ttt_linear_mini_batch_backward(
    # MatMul
    XQ_mini_batch,
    XK_mini_batch,
    W1_init,
    W1_last,
    # LnFusedL2BWD
    ln_weight,
    std_fused,
    x_hat_fused,
    grad_output_fused,
    grad_x_hat_fused,
    grad_l_wrt_Z1,
    # Dual Form
    last_eta_mini_batch,
    # LN
    std_ln,
    x_hat_ln,
    # Upstream gradients
    grad_L_W1_last,
    grad_L_b1_last,
    grad_L_XQW_mini_batch,
    # Constant expressions
    NH: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    mp_dtype,
):
    # Stage 4: LN
    grad_L_ln_weight_ln = tl.sum(grad_L_XQW_mini_batch * x_hat_ln, axis=0)
    grad_L_ln_bias_ln = tl.sum(grad_L_XQW_mini_batch, axis=0)

    grad_L_x_hat_ln = grad_L_XQW_mini_batch * ln_weight
    grad_L_Z1_bar = (
        (1.0 / F)
        * (
            F * grad_L_x_hat_ln
            - tl.sum(grad_L_x_hat_ln, axis=1)[:, None]
            - x_hat_ln * tl.sum(grad_L_x_hat_ln * x_hat_ln, axis=1)[:, None]
        )
        / std_ln
    )

    grad_L_W1_last += tl.dot(tl.trans(XQ_mini_batch).to(tl.bfloat16), grad_L_Z1_bar.to(tl.bfloat16))
    grad_L_b1_last += tl.sum(grad_L_Z1_bar, axis=0)

    grad_L_grad_l_wrt_Z1 = -(
        tl.dot((last_eta_mini_batch * XK_mini_batch).to(mp_dtype), grad_L_W1_last.to(mp_dtype))
    ) - (last_eta_mini_batch * grad_L_b1_last)

    grad_L_XQ_mini_batch = tl.dot(grad_L_Z1_bar.to(mp_dtype), tl.trans(W1_last).to(mp_dtype))

    grad_l_wrt_Z1_Last = tl.trans(tl.dot(grad_L_W1_last.to(mp_dtype), tl.trans(grad_l_wrt_Z1).to(mp_dtype)))

    grad_L_XK_mini_batch = -grad_l_wrt_Z1_Last * last_eta_mini_batch

    grad_L_last_eta_in_mini_batch = tl.sum(
        -(grad_l_wrt_Z1_Last * XK_mini_batch) - (grad_L_b1_last * grad_l_wrt_Z1), axis=1
    )[None, :]

    last_mini_batch_mask = tl.arange(0, CS)[:, None] == CS - 1
    grad_L_eta_mini_batch = tl.where(last_mini_batch_mask, grad_L_last_eta_in_mini_batch, 0)

    # Stage 2: LnFusedL2BWD
    grad_L_grad_x_hat_fused = (
        (1.0 / std_fused) * grad_L_grad_l_wrt_Z1
        + (1.0 / F) * tl.sum(-grad_L_grad_l_wrt_Z1 * (1.0 / std_fused), axis=1)[:, None]
        + (1.0 / F) * x_hat_fused * tl.sum(-grad_L_grad_l_wrt_Z1 * (1.0 / std_fused) * x_hat_fused, axis=1)[:, None]
    )

    grad_L_y = ln_weight * grad_L_grad_x_hat_fused

    grad_L_ln_weight_fused = tl.sum(grad_output_fused * grad_L_grad_x_hat_fused + grad_L_y * x_hat_fused, axis=0)
    grad_L_ln_bias_fused = tl.sum(grad_L_y, axis=0)

    grad_L_x_hat_fused = (
        grad_L_y * ln_weight
        + (1.0 / F)
        * grad_x_hat_fused
        * tl.sum(-grad_L_grad_l_wrt_Z1 * (1.0 / std_fused) * x_hat_fused, axis=1)[:, None]
        + (1.0 / F)
        * tl.sum(grad_x_hat_fused * x_hat_fused, axis=1)[:, None]
        * (-grad_L_grad_l_wrt_Z1 * (1.0 / std_fused))
    )

    grad_L_std = -grad_L_x_hat_fused * (x_hat_fused / std_fused) - (
        grad_L_grad_l_wrt_Z1 * ((grad_l_wrt_Z1 * std_fused) / (std_fused * std_fused))
    )

    grad_L_Z1 = (
        grad_L_x_hat_fused * (1.0 / std_fused)
        - (1.0 / F) * tl.sum(grad_L_x_hat_fused, axis=1)[:, None] * (1.0 / std_fused)
        + (1.0 / F) * tl.sum(grad_L_std, axis=1)[:, None] * x_hat_fused
    )

    grad_L_reconstruction_target = -ln_weight * grad_L_grad_x_hat_fused

    # Stage 1: MatMul
    grad_L_XQ = grad_L_XQW_mini_batch + grad_L_XQ_mini_batch
    grad_L_XV = grad_L_reconstruction_target

    grad_L_XK = (
        -grad_L_reconstruction_target
        + grad_L_XK_mini_batch
        + tl.dot(grad_L_Z1.to(mp_dtype), tl.trans(W1_init.to(mp_dtype)))
    )

    grad_L_W1_init = grad_L_W1_last + tl.dot(tl.trans(XK_mini_batch.to(tl.bfloat16)), grad_L_Z1.to(tl.bfloat16))
    grad_L_b1_init = grad_L_b1_last + (tl.sum(grad_L_Z1, axis=0))

    # NOTE: Sum over batch post-kernel to avoid sync barrier
    grad_L_ttt_norm_weight_mini_batch = (grad_L_ln_weight_ln + grad_L_ln_weight_fused)[None, :]
    grad_L_ttt_norm_bias_mini_batch = (grad_L_ln_bias_ln + grad_L_ln_bias_fused)[None, :]

    return (
        grad_L_ttt_norm_weight_mini_batch,
        grad_L_ttt_norm_bias_mini_batch,
        grad_L_W1_init,
        grad_L_b1_init,
        grad_L_XQ,
        grad_L_XV,
        grad_L_XK,
        grad_L_eta_mini_batch,
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=4),
    ],
    key=["checkpoint_group_size"],
)
@triton.jit
def ttt_linear_scan_backward(
    XQ_batch_ptr,
    XV_batch_ptr,
    XK_batch_ptr,
    eta_batch_ptr,
    ttt_norm_weight_ptr,
    ttt_norm_bias_ptr,
    W1_checkpoints_ptr,
    b1_checkpoints_ptr,
    # Upstream gradients
    grad_L_W1_last_ptr,
    grad_L_b1_last_ptr,
    grad_L_XQW_mini_batch_ptr,
    # Intermediate buffers
    W1_init_group_ptr,
    x_hat_ln_group_ptr,
    std_ln_group_ptr,
    grad_l_wrt_Z1_group_ptr,
    x_hat_fused_group_ptr,
    grad_x_hat_fused_group_ptr,
    grad_output_fused_group_ptr,
    std_fused_group_ptr,
    # Output buffers
    grad_L_ttt_norm_weight_ptr,
    grad_L_ttt_norm_bias_ptr,
    grad_L_W1_init_ptr,
    grad_L_b1_init_ptr,
    grad_L_XQ_ptr,
    grad_L_XV_ptr,
    grad_L_XK_ptr,
    grad_L_eta_ptr,
    # Strides
    CS_F_stride: tl.constexpr,
    F_F_stride: tl.constexpr,
    CS_CS_stride: tl.constexpr,
    F_stride: tl.constexpr,
    # Constant expressions
    NH: tl.constexpr,
    NC: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    K: tl.constexpr,
    checkpoint_group_size: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    mp_dtype = XQ_batch_ptr.type.element_ty

    K_F_F_stride = K * F * F
    K_F_stride = K * F
    CS_stride = CS

    F_F_offset = batch * NH * F_F_stride + head * F_F_stride + tl.arange(0, F)[:, None] * F + tl.arange(0, F)[None, :]
    F_offset = batch * NH * F_stride + head * F_stride + tl.arange(0, F)[None, :]
    norm_offset = head * F_stride + tl.arange(0, F)
    norm_store_offset = batch * NH * F_stride + head * F_stride + tl.arange(0, F)[None, :]

    ln_weight = tl.load(ttt_norm_weight_ptr + norm_offset).to(tl.float32)[None, :]
    ln_bias = tl.load(ttt_norm_bias_ptr + norm_offset).to(tl.float32)[None, :]

    # Load upstream gradients
    grad_L_W1_last = tl.load(grad_L_W1_last_ptr + F_F_offset).to(tl.float32)
    grad_L_b1_last = tl.load(grad_L_b1_last_ptr + F_offset).to(tl.float32)

    # Allocate stack for accumulated output gradients
    grad_L_ttt_norm_weight = tl.zeros((1, F), dtype=tl.float32)
    grad_L_ttt_norm_bias = tl.zeros((1, F), dtype=tl.float32)

    # Iterate over checkpoints in reverse
    for checkpoint_idx in range(K - 1, -1, -1):
        W1_checkpoint_offset = (
            batch * NH * K_F_F_stride
            + head * K_F_F_stride
            + checkpoint_idx * F_F_stride
            + tl.arange(0, F)[:, None] * F
            + tl.arange(0, F)[None, :]
        )
        b1_checkpoint_offset = (
            batch * NH * K_F_stride
            + head * K_F_stride
            + checkpoint_idx * F_stride
            + tl.arange(0, 1)[:, None] * F
            + tl.arange(0, F)[None, :]
        )

        W1_init = tl.load(W1_checkpoints_ptr + W1_checkpoint_offset).to(tl.float32)
        b1_init = tl.load(b1_checkpoints_ptr + b1_checkpoint_offset).to(tl.float32)

        # Forward over mini-batches in checkpoint group
        for mini_batch_idx_in_group in range(0, checkpoint_group_size):
            mini_batch_idx = checkpoint_idx * checkpoint_group_size + mini_batch_idx_in_group

            # Edge case: checkpoint group size isnt perfectly divisible by NC
            if mini_batch_idx < NC:

                CS_F_offset = (
                    batch * NH * NC * CS_F_stride
                    + head * NC * CS_F_stride
                    + mini_batch_idx * CS_F_stride
                    + tl.arange(0, CS)[:, None] * F
                    + tl.arange(0, F)[None, :]
                )
                CS_CS_offset = (
                    batch * NH * NC * CS_CS_stride
                    + head * NC * CS_CS_stride
                    + mini_batch_idx * CS_CS_stride
                    + tl.arange(0, CS)[:, None] * CS
                    + tl.arange(0, CS)[None, :]
                )
                last_CS_offset = (
                    batch * NH * NC * CS_CS_stride
                    + head * NC * CS_CS_stride
                    + mini_batch_idx * CS_CS_stride
                    + (CS - 1) * CS
                    + tl.arange(0, CS)[:, None]
                )

                XQ_mini_batch = tl.load(XQ_batch_ptr + CS_F_offset).to(tl.float32)
                XV_mini_batch = tl.load(XV_batch_ptr + CS_F_offset).to(tl.float32)
                XK_mini_batch = tl.load(XK_batch_ptr + CS_F_offset).to(tl.float32)
                last_eta_mini_batch = tl.load(eta_batch_ptr + last_CS_offset).to(tl.float32)

                (
                    XQW_mini_batch,
                    W1_curr,
                    b1_curr,
                    x_hat_ln,
                    std_ln,
                    grad_l_wrt_Z1,
                    x_hat_fused,
                    grad_x_hat_fused,
                    grad_output_fused,
                    std_fused,
                ) = ttt_linear_mini_batch_forward(
                    W1_init,
                    b1_init,
                    ln_weight,
                    ln_bias,
                    XQ_mini_batch,
                    XK_mini_batch,
                    XV_mini_batch,
                    last_eta_mini_batch,
                    CS,
                    F,
                    mp_dtype,
                )

                G_CS_F_offset = (
                    batch * NH * checkpoint_group_size * CS_F_stride
                    + head * checkpoint_group_size * CS_F_stride
                    + mini_batch_idx_in_group * CS_F_stride
                    + tl.arange(0, CS)[:, None] * F
                    + tl.arange(0, F)[None, :]
                )
                G_F_F_offset = (
                    batch * NH * checkpoint_group_size * F_F_stride
                    + head * checkpoint_group_size * F_F_stride
                    + mini_batch_idx_in_group * F_F_stride
                    + tl.arange(0, F)[:, None] * F
                    + tl.arange(0, F)[None, :]
                )
                G_CS_offset = (
                    batch * NH * checkpoint_group_size * CS_stride
                    + head * checkpoint_group_size * CS_stride
                    + mini_batch_idx_in_group * CS_stride
                    + tl.arange(0, CS)[:, None]
                    + tl.arange(0, 1)[None, :]
                )

                # Store intermediate values
                tl.store(W1_init_group_ptr + G_F_F_offset, W1_init)
                tl.store(x_hat_ln_group_ptr + G_CS_F_offset, x_hat_ln)
                tl.store(std_ln_group_ptr + G_CS_offset, std_ln)
                tl.store(grad_l_wrt_Z1_group_ptr + G_CS_F_offset, grad_l_wrt_Z1)
                tl.store(x_hat_fused_group_ptr + G_CS_F_offset, x_hat_fused)
                tl.store(grad_x_hat_fused_group_ptr + G_CS_F_offset, grad_x_hat_fused)
                tl.store(grad_output_fused_group_ptr + G_CS_F_offset, grad_output_fused)
                tl.store(std_fused_group_ptr + G_CS_offset, std_fused)

                W1_init = W1_curr
                b1_init = b1_curr

        W1_last = W1_init  # For backward

        # Backward over mini-batches in checkpoint group in reverse
        for mini_batch_idx_in_group in range(checkpoint_group_size - 1, -1, -1):
            mini_batch_idx = checkpoint_idx * checkpoint_group_size + mini_batch_idx_in_group
            if mini_batch_idx < NC:

                CS_F_offset = (
                    batch * NH * NC * CS_F_stride
                    + head * NC * CS_F_stride
                    + mini_batch_idx * CS_F_stride
                    + tl.arange(0, CS)[:, None] * F
                    + tl.arange(0, F)[None, :]
                )
                CS_CS_offset = (
                    batch * NH * NC * CS_CS_stride
                    + head * NC * CS_CS_stride
                    + mini_batch_idx * CS_CS_stride
                    + tl.arange(0, CS)[:, None] * CS
                    + tl.arange(0, CS)[None, :]
                )
                last_CS_offset = (
                    batch * NH * NC * CS_CS_stride
                    + head * NC * CS_CS_stride
                    + mini_batch_idx * CS_CS_stride
                    + (CS - 1) * CS
                    + tl.arange(0, CS)[:, None]
                )

                G_CS_F_offset = (
                    batch * NH * checkpoint_group_size * CS_F_stride
                    + head * checkpoint_group_size * CS_F_stride
                    + mini_batch_idx_in_group * CS_F_stride
                    + tl.arange(0, CS)[:, None] * F
                    + tl.arange(0, F)[None, :]
                )
                G_F_F_offset = (
                    batch * NH * checkpoint_group_size * F_F_stride
                    + head * checkpoint_group_size * F_F_stride
                    + mini_batch_idx_in_group * F_F_stride
                    + tl.arange(0, F)[:, None] * F
                    + tl.arange(0, F)[None, :]
                )
                G_CS_offset = (
                    batch * NH * checkpoint_group_size * CS_stride
                    + head * checkpoint_group_size * CS_stride
                    + mini_batch_idx_in_group * CS_stride
                    + tl.arange(0, CS)[:, None]
                    + tl.arange(0, 1)[None, :]
                )

                XQ_mini_batch = tl.load(XQ_batch_ptr + CS_F_offset)
                XK_mini_batch = tl.load(XK_batch_ptr + CS_F_offset).to(tl.float32)

                grad_L_XQW_mini_batch = tl.load(grad_L_XQW_mini_batch_ptr + CS_F_offset).to(tl.float32)

                # Inputs
                last_eta_mini_batch = tl.load(eta_batch_ptr + last_CS_offset).to(tl.float32)

                # Remated values
                W1_curr = tl.load(W1_init_group_ptr + G_F_F_offset).to(tl.float32)

                x_hat_ln = tl.load(x_hat_ln_group_ptr + G_CS_F_offset).to(tl.float32)
                std_ln = tl.load(std_ln_group_ptr + G_CS_offset).to(tl.float32)
                grad_l_wrt_Z1 = tl.load(grad_l_wrt_Z1_group_ptr + G_CS_F_offset).to(tl.float32)
                grad_output_fused = tl.load(grad_output_fused_group_ptr + G_CS_F_offset).to(tl.float32)
                std_fused = tl.load(std_fused_group_ptr + G_CS_offset).to(tl.float32)

                # Something about these 2 loads causes time to double if casted to float32.
                # We leave them in BF16 for now, even though it will cause slightly less precise results.
                x_hat_fused = tl.load(x_hat_fused_group_ptr + G_CS_F_offset).to(mp_dtype)
                grad_x_hat_fused = tl.load(grad_x_hat_fused_group_ptr + G_CS_F_offset).to(mp_dtype)

                (
                    grad_L_ttt_norm_weight_mini_batch,
                    grad_L_ttt_norm_bias_mini_batch,
                    grad_L_W1_init,
                    grad_L_b1_init,
                    grad_L_XQ,
                    grad_L_XV,
                    grad_L_XK,
                    grad_L_eta_mini_batch,
                ) = ttt_linear_mini_batch_backward(
                    # MatMul
                    XQ_mini_batch,
                    XK_mini_batch,
                    W1_curr,
                    W1_last,
                    # LnFusedL2BWD
                    ln_weight,
                    std_fused,
                    x_hat_fused,
                    grad_output_fused,
                    grad_x_hat_fused,
                    grad_l_wrt_Z1,
                    # Dual Form
                    last_eta_mini_batch,
                    # LN
                    std_ln,
                    x_hat_ln,
                    # Upstream gradients
                    grad_L_W1_last,
                    grad_L_b1_last,
                    grad_L_XQW_mini_batch,
                    # Constant expressions
                    NH,
                    CS,
                    F,
                    mp_dtype,
                )

                # Store mini-batch output gradients
                tl.store(grad_L_XQ_ptr + CS_F_offset, grad_L_XQ)
                tl.store(grad_L_XV_ptr + CS_F_offset, grad_L_XV)
                tl.store(grad_L_XK_ptr + CS_F_offset, grad_L_XK)
                tl.store(grad_L_eta_ptr + CS_CS_offset, grad_L_eta_mini_batch)

                # Accumulate / update output gradients
                grad_L_W1_last = grad_L_W1_init
                grad_L_b1_last = grad_L_b1_init
                grad_L_ttt_norm_weight += grad_L_ttt_norm_weight_mini_batch
                grad_L_ttt_norm_bias += grad_L_ttt_norm_bias_mini_batch

                # Update W1_last to be the init state as we move a step backwards
                W1_last = W1_curr

    tl.store(grad_L_ttt_norm_weight_ptr + norm_store_offset, grad_L_ttt_norm_weight)
    tl.store(grad_L_ttt_norm_bias_ptr + norm_store_offset, grad_L_ttt_norm_bias)
    tl.store(grad_L_W1_init_ptr + F_F_offset, grad_L_W1_last)
    tl.store(grad_L_b1_init_ptr + F_offset, grad_L_b1_last)
