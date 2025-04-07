import math
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import wandb
from einops import rearrange, repeat
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from tqdm import tqdm


def get_interleave_offsets(num_frames, num_chunks, tokens_per_frame, text_length):
    frames_per_chunk = num_frames // num_chunks

    base_offset, init_offset = frames_per_chunk, frames_per_chunk + (num_frames % frames_per_chunk)
    base_offset *= tokens_per_frame
    init_offset *= tokens_per_frame

    base_offset += text_length
    init_offset += text_length

    return base_offset, init_offset


@torch.compiler.disable
def to_local(tensor: torch.Tensor | DTensor) -> torch.Tensor | DTensor:
    """
    Convert a distributed tensor to a local tensor
    """
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


@torch.compiler.disable
def place_into(local_tensor: torch.Tensor, dt: DTensor | torch.Tensor) -> torch.Tensor | DTensor:
    """
    Place a local tensor into a distributed tensor with the same shape and placement
    """
    if not isinstance(dt, DTensor):
        return local_tensor

    return DTensor.from_local(
        local_tensor, device_mesh=dt.device_mesh, placements=dt.placements, shape=dt.shape, stride=dt.stride()
    )


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


def dropout_txt(txt_emb, p):
    txt_emb = (
        expand_dims_like(
            torch.bernoulli((1.0 - p) * torch.ones(txt_emb.shape[0], device=txt_emb.device)),
            txt_emb,
        )
        * txt_emb
    )
    return txt_emb


# sat/dit_ssm_video_concat.py:544
def modulate(x, shift, scale):
    while x.ndim != shift.ndim:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)

    return x * (1 + scale) + shift


# sat/sgm/util.py:274
def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


# sat/sgm/util.py:278
def append_dims(x, target_dims):
    """Appends singleton dimensions to the end of a tensor until it reaches target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"Cannot reduce dimensions: input has {x.ndim} dims but target_dims is {target_dims}")
    return x[(...,) + (None,) * dims_to_append]


# sat/sgm/util.py:417
def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)

    x1, x2 = to_local(x).unbind(dim=-1)
    x = place_into(torch.stack((-x2, x1), dim=-1), x)

    return rearrange(x, "... d r -> ... (d r)")


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False, dtype=torch.float32):
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=timesteps.device
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding.to(dtype)


# sat/dit_ssm_video_concat.py:398
def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


# sat/sgm/modules/diffusionmodules/discretizer.py:11
def generate_roughly_equally_spaced_steps(num_substeps, max_step):
    return np.linspace(max_step - 1, 0, num_substeps, endpoint=False).astype(int)[::-1]


# sat/sgm/modules/diffusionmodules/util.py:20
def make_beta_schedule(
    schedule,
    n_timestep,
    linear_start=1e-4,
    linear_end=2e-2,
):
    if schedule == "linear":
        betas = torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64) ** 2
    return betas.numpy()


# sat/dit_ssm_video_concat.py:549
def unpatchify(x, c, p, w, h, rope_position_ids=None, **kwargs):
    """
    x: (N, T/2 * S, patch_size**3 * C)
    imgs: (N, T, H, W, C)
    """
    if rope_position_ids is not None:
        assert NotImplementedError
        # do pix2struct unpatchify
        L = x.shape[1]
        x = x.reshape(shape=(x.shape[0], L, p, p, c))
        x = torch.einsum("nlpqc->ncplq", x)
        imgs = x.reshape(shape=(x.shape[0], c, p, L * p))
    else:
        b = x.shape[0]
        imgs = rearrange(x, "b (t h w) (c p q) -> b t c (h p) (w q)", b=b, h=h, w=w, c=c, p=p, q=p)

    return imgs


@torch.compiler.disable
def full_tensor(tensor: torch.Tensor | DTensor) -> torch.Tensor:
    """
    Convert a DTensor to a local replicalated tensor
    """
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()

    return tensor


def replicate_tensor(tensor: torch.Tensor | DTensor, tp_mesh):
    """
    Convert a tensor to a replicated DTensor
    """
    if tp_mesh is None:
        return tensor

    if not isinstance(tensor, DTensor):
        tensor = DTensor.from_local(tensor, tp_mesh, (Replicate(),), run_check=True)

    return tensor.redistribute(placements=(Replicate(),))


def shard_tensor(tensor: torch.Tensor | DTensor, tp_mesh=None, dim=0) -> torch.Tensor:
    """
    Shard a tensor along a dimension
    """
    if tp_mesh is None:
        return tensor

    if not isinstance(tensor, DTensor):
        tensor = DTensor.from_local(tensor, tp_mesh, (Replicate(),), run_check=True)

    return tensor.redistribute(placements=(Shard(dim),))


def cast_rotary_freqs(model, dtype):
    for _, module in model.named_modules():
        if isinstance(module, Rotary3DPositionEmbedding):
            for buffer_name, buffer in module.named_buffers():
                if buffer_name in ("freqs_cos", "freqs_sin"):
                    buffer.data = buffer.data.to(dtype=dtype)


@dataclass
class SequenceMetadata:
    """Dataclass for sequence information shared through forward pass."""

    text_length: int
    seq_text_length: int
    num_frames: int
    num_chunks: int
    tokens_per_frame: int
    latent_height: int
    latent_width: int
    t_emb: torch.Tensor

    # Multiscene interleave offset metadata
    base_offset: Optional[int] = None
    init_offset: Optional[int] = None

    @property
    def is_multiscene(self) -> bool:
        return self.num_chunks > 1

    def init_multiscene_offsets(self):
        """Set interleave offsets for multi-chunk processing."""

        self.base_offset, self.init_offset = get_interleave_offsets(
            num_frames=self.num_frames,
            num_chunks=self.num_chunks,
            tokens_per_frame=self.tokens_per_frame,
            text_length=self.text_length,
        )


# sat/sgm/modules/diffusionmodules/denoiser_scaling.py:52
class VideoScaling:
    def __call__(self, sigma, idx):
        c_skip = sigma
        c_out = -((1 - sigma**2) ** 0.5)
        c_in = torch.ones_like(sigma, device=sigma.device)
        c_noise = idx.clone()
        return c_skip, c_out, c_in, c_noise


# sat/sgm/modules/diffusionmodules/sigma_sampling.py:19
class DiscreteSampler:
    def __init__(self, config, effective_rank, effective_world_size, uniform_sampling=True):
        self.sigma_interval = config.sigma_interval
        self.sigmas = None
        self.uniform_sampling = uniform_sampling

        if not dist.is_initialized():
            return  

        self.effective_rank = effective_rank

        if self.uniform_sampling:
            num_idx = self.sigma_interval  # treat sigma_interval as total indices
            i = 1
            while True:
                if effective_world_size % i != 0 or num_idx % (effective_world_size // i) != 0:
                    i += 1
                else:
                    self.group_num = effective_world_size // i
                    break

            self.group_width = effective_world_size // self.group_num
            self.group_sigma_interval = num_idx // self.group_num

    def __call__(self, n_samples, rand=None, return_idx=True, generator=None, device="cuda"):
        # Lazy initialization of sigmas to avoid 'meta' device issues
        if self.sigmas is None:
            self.sigmas = ZeroSNRDDPMDiscretization()(self.sigma_interval, device=device, flip=True)

        if self.uniform_sampling:
            group_index = self.effective_rank // self.group_width
            start = group_index * self.group_sigma_interval
            end = (group_index + 1) * self.group_sigma_interval
            if rand is None:
                idx = torch.randint(start, end, (n_samples,), generator=generator, device=device)
            else:
                idx = torch.full((n_samples,), rand, dtype=torch.long).to(device)
        else:
            if rand is None:
                idx = torch.randint(0, self.sigma_interval, (n_samples,), generator=generator, device=device)
            else:
                idx = torch.full((n_samples,), rand, dtype=torch.long).to(device)

        if return_idx:
            return self.sigmas[idx], idx
        else:
            return self.sigmas[idx]


# sat/sgm/modules/diffusionmodules/discretizer.py:74
class ZeroSNRDDPMDiscretization:
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
        shift_scale=1.0,
    ):
        self.num_timesteps = num_timesteps
        betas = make_beta_schedule("linear", num_timesteps, linear_start, linear_end)
        alphas = 1.0 - betas

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod = self.alphas_cumprod / (shift_scale + (1 - shift_scale) * self.alphas_cumprod)

    def get_sigmas(self, n, device="cuda", return_idx=False):
        if n < self.num_timesteps:
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        alphas_cumprod = to_torch(alphas_cumprod)
        alphas_cumprod_sqrt = alphas_cumprod.sqrt()  # Sqrt causes mismatch
        alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
        alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()

        alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
        alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / (alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T)

        if return_idx:
            return torch.flip(alphas_cumprod_sqrt, (0,)), timesteps
        else:
            return torch.flip(alphas_cumprod_sqrt, (0,))

    def __call__(self, n, do_append_zero=False, device="cuda", flip=False, return_idx=False):
        if return_idx:
            sigmas, idx = self.get_sigmas(n, device=device, return_idx=return_idx)
        else:
            sigmas = self.get_sigmas(n, device=device, return_idx=return_idx)
        sigmas = append_zero(sigmas) if do_append_zero else sigmas
        if return_idx:
            return sigmas if not flip else torch.flip(sigmas, (0,)), idx
        else:
            return sigmas if not flip else torch.flip(sigmas, (0,))


# sat/dit_ssm_video_concat.py:424
class Rotary3DPositionEmbedding(nn.Module):
    def __init__(
        self,
        height,
        width,
        compressed_num_frames,
        head_dim,
        theta=10000,
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.compressed_num_frames = compressed_num_frames
        self.head_dim = head_dim
        self.theta = theta
        self.tp_mesh = None

        freqs_sin, freqs_cos = self._precompute_freqs_cis()

        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)

    def init_device_mesh(self, tp_mesh):
        self.tp_mesh = tp_mesh

    def _precompute_freqs_cis(self):
        dim_t = self.head_dim // 4
        dim_h = self.head_dim // 8 * 3
        dim_w = self.head_dim // 8 * 3

        freqs_t = 1.0 / (self.theta ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
        freqs_h = 1.0 / (self.theta ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
        freqs_w = 1.0 / (self.theta ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))

        grid_t = torch.arange(self.compressed_num_frames, dtype=torch.float32)
        grid_h = torch.arange(self.height, dtype=torch.float32)
        grid_w = torch.arange(self.width, dtype=torch.float32)

        freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)

        freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

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

        return freqs.sin(), freqs.cos()

    def init_freqs(self):
        freqs_sin, freqs_cos = self._precompute_freqs_cis()
        self.freqs_sin.copy_(freqs_sin)
        self.freqs_cos.copy_(freqs_cos)

        if self.tp_mesh is not None:
            self.freqs_sin = replicate_tensor(self.freqs_sin, self.tp_mesh)
            self.freqs_cos = replicate_tensor(self.freqs_cos, self.tp_mesh)

    def forward(self, t):
        seq_len = t.shape[2]
        freqs_cos = self.freqs_cos[:seq_len].unsqueeze(0).unsqueeze(0)
        freqs_sin = self.freqs_sin[:seq_len].unsqueeze(0).unsqueeze(0)

        return t * freqs_cos + rotate_half(t) * freqs_sin


# sat/sgm/modules/diffusionmodules/denoiser.py:41
class DiscreteDenoiser(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        num_idx: int,
        dtype,
        do_append_zero=False,
        quantize_c_noise=True,
        flip=True,
    ):
        super().__init__()

        self.scaling = VideoScaling()

        self.sigmas = ZeroSNRDDPMDiscretization()(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.quantize_c_noise = quantize_c_noise
        self.network = network
        self.dtype = dtype

    def forward(
        self,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: Dict,
        **additional_model_inputs,
    ) -> torch.Tensor:
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma, **additional_model_inputs)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        scaled = (input * c_in).to(dtype=self.dtype)

        # Process each batch separately and collect results
        batch_size = input.shape[0]
        results = []

        for i in range(batch_size):
            # Extract single batch elements
            batch_scaled = scaled[i : i + 1]
            batch_crossattn = cond["crossattn"][i : i + 1]
            batch_c_noise = c_noise[i : i + 1]

            # Forward pass for this batch element
            network_output = self.network(batch_scaled, batch_crossattn, batch_c_noise)

            # Apply scaling factors
            batch_result = network_output * c_out[i : i + 1] + input[i : i + 1] * c_skip[i : i + 1]
            results.append(batch_result)

        # Concatenate results along batch dimension
        return torch.cat(results, dim=0)

    def sigma_to_idx(self, sigma):
        dists = sigma - self.sigmas.to(sigma.device)[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx):
        return self.sigmas.to(idx.device)[idx]

    def possibly_quantize_sigma(self, sigma):
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise):
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise


# sat/sgm/modules/diffusionmodules/sampling_utils.py:8
class NoDynamicThresholding:
    def __call__(self, uncond, cond, scale):
        scale = append_dims(scale, cond.ndim) if isinstance(scale, torch.Tensor) else scale
        return uncond + scale * (cond - uncond)


# sat/sgm/modules/diffusionmodules/guiders.py:56
class DynamicCFG:
    def __init__(self, scale, exp, num_steps):
        scale_schedule = (
            lambda scale, sigma, step_index: 1 + scale * (1 - math.cos(math.pi * (step_index / num_steps) ** exp)) / 2
        )

        self.scale_schedule = partial(scale_schedule, scale)
        self.dyn_thresh = NoDynamicThresholding()

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat"]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out

    def __call__(self, x, sigma, step_index, scale=None):
        x_u, x_c = x.chunk(2)
        scale_value = self.scale_schedule(sigma, step_index.item())
        x_pred = self.dyn_thresh(x_u, x_c, scale_value)
        return x_pred


# sat/sgm/modules/diffusionmodules/sampling.py:569
class VPSDEDPMPP2MSampler:
    def __init__(
        self,
        denoiser: nn.Module,
        discretization_config: Dict,
        num_steps: int,
        guider_config: Dict,
        use_wandb: bool,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.denoiser = denoiser
        self.num_steps = num_steps
        self.discretization = ZeroSNRDDPMDiscretization(**discretization_config)
        self.guider = DynamicCFG(**guider_config)
        self.verbose = verbose
        self.device = device
        self.use_wandb = use_wandb

    def denoise(self, x, alpha_cumprod_sqrt, cond, uc, timestep=None, idx=None, scale=None, scale_emb=None):
        additional_model_inputs = {}

        if not isinstance(scale, torch.Tensor) and scale == 1:
            additional_model_inputs["idx"] = x.new_ones([x.shape[0]]) * timestep
            if scale_emb is not None:
                additional_model_inputs["scale_emb"] = scale_emb
            denoised = self.denoiser(x, alpha_cumprod_sqrt, cond, **additional_model_inputs).to(torch.float32)
        else:
            additional_model_inputs["idx"] = torch.cat([x.new_ones([x.shape[0]]) * timestep] * 2)
            denoised = self.denoiser(
                *self.guider.prepare_inputs(x, alpha_cumprod_sqrt, cond, uc), **additional_model_inputs
            ).to(dtype=torch.float32)
            if isinstance(self.guider, DynamicCFG):
                denoised = self.guider(
                    denoised, (1 - alpha_cumprod_sqrt**2) ** 0.5, step_index=self.num_steps - timestep, scale=scale
                )
            else:
                denoised = self.guider(denoised, (1 - alpha_cumprod_sqrt**2) ** 0.5, scale=scale)
        return denoised

    def get_variables(self, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt=None):
        alpha_cumprod = alpha_cumprod_sqrt**2
        lamb = ((alpha_cumprod / (1 - alpha_cumprod)) ** 0.5).log()
        next_alpha_cumprod = next_alpha_cumprod_sqrt**2
        lamb_next = ((next_alpha_cumprod / (1 - next_alpha_cumprod)) ** 0.5).log()
        h = lamb_next - lamb

        if previous_alpha_cumprod_sqrt is not None:
            previous_alpha_cumprod = previous_alpha_cumprod_sqrt**2
            lamb_previous = ((previous_alpha_cumprod / (1 - previous_alpha_cumprod)) ** 0.5).log()
            h_last = lamb - lamb_previous
            r = h_last / h
            return h, r, lamb, lamb_next
        else:
            return h, None, lamb, lamb_next

    def get_mult(self, h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt):
        mult1 = ((1 - next_alpha_cumprod_sqrt**2) / (1 - alpha_cumprod_sqrt**2)) ** 0.5 * (-h).exp()
        mult2 = (-2 * h).expm1() * next_alpha_cumprod_sqrt

        if previous_alpha_cumprod_sqrt is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        alpha_cumprod_sqrt, timesteps = self.discretization(
            self.num_steps if num_steps is None else num_steps,
            device=self.device,
            return_idx=True,
            do_append_zero=False,
        )
        alpha_cumprod_sqrt = torch.cat([alpha_cumprod_sqrt, alpha_cumprod_sqrt.new_ones([1])])
        timesteps = torch.cat([torch.tensor(list(timesteps)).new_zeros([1]) - 1, torch.tensor(list(timesteps))])

        uc = uc or cond

        num_sigmas = len(alpha_cumprod_sqrt)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps

    def sampler_step(
        self,
        old_denoised,
        previous_alpha_cumprod_sqrt,
        alpha_cumprod_sqrt,
        next_alpha_cumprod_sqrt,
        x,
        cond,
        uc=None,
        idx=None,
        timestep=None,
    ):
        denoised = self.denoise(x, alpha_cumprod_sqrt, cond, uc, timestep, idx).to(torch.float32)
        if idx == 1:
            return denoised, denoised

        h, r, lamb, lamb_next = self.get_variables(
            alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt
        )
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt)
        ]

        mult_noise = append_dims((1 - next_alpha_cumprod_sqrt**2) ** 0.5 * (1 - (-2 * h).exp()) ** 0.5, x.ndim)
        x_standard = mult[0] * x - mult[1] * denoised + mult_noise * torch.randn_like(x)
        if old_denoised is None or torch.sum(next_alpha_cumprod_sqrt) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d + mult_noise * torch.randn_like(x)

            x = x_advanced

        return x, denoised

    def __call__(self, x, cond, uc=None, num_steps=None, scale=None, **kwargs):
        x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * alpha_cumprod_sqrt[i - 1],
                s_in * alpha_cumprod_sqrt[i],
                s_in * alpha_cumprod_sqrt[i + 1],
                x,
                cond,
                uc=uc,
                idx=self.num_steps - i,
                timestep=timesteps[-(i + 1)],
            )

            if self.use_wandb and wandb.run is not None:
                wandb.log(
                    {
                        "denoising_progress/step": i + 1,
                        "denoising_progress/total_steps": num_sigmas,
                    },
                    step=i + 1,
                )

        return x
