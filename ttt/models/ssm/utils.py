from typing import Tuple

import torch
from einops import rearrange

from ttt.models.cogvideo.utils import broadcat


def precompute_freqs_cis_3d(
    dim: int, height: int, width: int, compressed_num_frames: int, theta: float = 10000.0
) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.
    Args:
        dim (int): Dimension of the frequency tensor.
        height (int): Height of the latents feature map.
        width (int): Width of the latents feature map.
        compressed_number_frames (int): Number of frames of the latents feature map.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    dim_t = dim // 4
    dim_h = dim // 8 * 3
    dim_w = dim // 8 * 3
    freqs_t = 1.0 / (theta ** (torch.arange(0, dim_t, 2)[: dim_t // 2].float() / dim_t))
    freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2)[: dim_h // 2].float() / dim_h))
    freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2)[: dim_w // 2].float() / dim_w))

    grid_t = torch.arange(compressed_num_frames, dtype=torch.float32)
    grid_h = torch.arange(height, dtype=torch.float32)
    grid_w = torch.arange(width, dtype=torch.float32)

    freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
    freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
    freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)

    freqs = broadcat(
        (
            freqs_t[:, None, None, :],
            freqs_h[None, :, None, :],
            freqs_w[None, None, :, :],
        ),
        dim=-1,
    )

    freqs = rearrange(freqs, "t h w d -> (t h w) d")
    freqs = freqs.contiguous()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1]), f"{freqs_cis.shape} != {seqlen, x.shape[-1]}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def scan(f, init, xs, checkpoint_group=0):
    """Mimic jax.lax.scan function."""
    carry = init
    if isinstance(xs, dict):
        num_items = len(next(iter(xs.values())))
    else:
        num_items = len(xs[0])

    def scan_fn(carry, i_start, i_end):
        sub_out_list = []
        for i in range(i_start, i_end):
            if isinstance(xs, dict):
                x = {key: tensor[i] for key, tensor in xs.items()}
            else:
                x = [x[i] for x in xs]
            carry, y = f(carry, x)
            sub_out_list.append(y)
        sub_out = torch.stack(sub_out_list)
        return carry, sub_out

    if checkpoint_group > 0:
        out_list = []
        for k in range(0, num_items, checkpoint_group):
            carry, sub_out = torch.utils.checkpoint.checkpoint(
                scan_fn,
                carry,
                k,
                min(k + checkpoint_group, num_items),
                use_reentrant=False,
            )
            out_list.append(sub_out)
        out = torch.concatenate(out_list, dim=0)
    else:
        carry, out = scan_fn(carry, 0, num_items)

    return carry, out
