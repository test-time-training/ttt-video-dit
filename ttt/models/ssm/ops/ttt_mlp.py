import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map

from ttt.models.ssm.ops.utils import gelu_bwd, ln_fused_l2_bwd, ln_fwd
from ttt.models.ssm.utils import scan


@torch.compile
def compute_mini_batch(params_dict, inputs):
    W1_init = params_dict["W1_states"]
    b1_init = params_dict["b1_states"]
    W2_init = params_dict["W2_states"]
    b2_init = params_dict["b2_states"]

    ttt_norm_weight = params_dict["ttt_norm_weight"]
    ttt_norm_bias = params_dict["ttt_norm_bias"]

    XQ_mini_batch = inputs["XQ"]
    XV_mini_batch = inputs["XV"]
    XK_mini_batch = inputs["XK"]

    eta_mini_batch = inputs["eta"]

    num_heads = XQ_mini_batch.size(1)
    head_dim = XQ_mini_batch.size(-1)

    X1 = XK_mini_batch
    Z1 = X1 @ W1_init + b1_init
    X2 = F.gelu(Z1, approximate="tanh")
    Z2 = X2 @ W2_init + b2_init
    reconstruction_target = XV_mini_batch - XK_mini_batch

    ln_weight = ttt_norm_weight.reshape(num_heads, 1, head_dim)
    ln_bias = ttt_norm_bias.reshape(num_heads, 1, head_dim)
    grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, reconstruction_target, ln_weight, ln_bias)
    grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-2, -1) * gelu_bwd(Z1)

    Attn1 = XQ_mini_batch @ X1.transpose(-2, -1)
    b1_bar = b1_init - eta_mini_batch @ grad_l_wrt_Z1
    Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar
    X2_bar = F.gelu(Z1_bar, approximate="tanh")

    Attn2 = X2_bar @ X2.transpose(-2, -1)
    b2_bar = b2_init - eta_mini_batch @ grad_l_wrt_Z2
    Z2_bar = X2_bar @ W2_init - (eta_mini_batch * Attn2) @ grad_l_wrt_Z2 + b2_bar

    last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
    W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
    b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
    W2_last = W2_init - (last_eta_mini_batch * X2).transpose(-1, -2) @ grad_l_wrt_Z2
    b2_last = b2_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z2, dim=-2, keepdim=True)

    Z2_bar = ln_fwd(Z2_bar, ln_weight, ln_bias)

    XQW_mini_batch = XQ_mini_batch + Z2_bar

    last_param_dict = {
        "W1_states": W1_last,
        "b1_states": b1_last,
        "W2_states": W2_last,
        "b2_states": b2_last,
        "ttt_norm_weight": ttt_norm_weight,
        "ttt_norm_bias": ttt_norm_bias,
    }

    return last_param_dict, XQW_mini_batch


def ttt_mlp(XK, XQ, XV, eta, ttt_norm_weight, ttt_norm_bias, W1_init, b1_init, W2_init, b2_init, checkpoint_group_size):
    init_params_dict = {
        "W1_states": W1_init,
        "b1_states": b1_init,
        "W2_states": W2_init,
        "b2_states": b2_init,
        "ttt_norm_weight": ttt_norm_weight,
        "ttt_norm_bias": ttt_norm_bias,
    }

    inputs = {
        "XK": XK,
        "XQ": XQ,
        "XV": XV,
        "eta": eta,
    }

    # Reorder such that mini-batch is first dimension for iteration
    inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

    XQW_batch = torch.empty_like(inputs["XK"])

    _, XQW_batch = scan(
        compute_mini_batch,  # Function to iterate over
        init_params_dict,
        inputs,
        checkpoint_group_size,
    )

    return XQW_batch.permute(1, 0, 3, 2, 4)
