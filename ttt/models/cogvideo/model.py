import torch
import torch.nn as nn

from ttt.models.cogvideo.dit import DiffusionTransformer
from ttt.models.cogvideo.utils import DiscreteSampler, VideoScaling, append_dims


class CogVideoX(nn.Module):
    def __init__(self, config, effective_rank: int, effective_world_size: int):
        super().__init__()
        self.config = config

        self.sigma_sampler = DiscreteSampler(config, effective_rank, effective_world_size)
        self.scaling = VideoScaling()

        self.dit = DiffusionTransformer(config)

        self.effective_rank = effective_rank
        self.tp_mesh = None

        self.noise_generator = None

    def init_ssm_weights(self):
        for layer in self.dit.layers:
            layer.seq_modeling_block.rotary.init_freqs()  # type: ignore
            layer.seq_modeling_block.ssm.init_freqs()  # type: ignore
            layer.seq_modeling_block.ssm.ttt.init_weights()  # type: ignore

    def setup_generator(self, seed: int):
        self.noise_generator = torch.Generator(device="cuda")
        self.noise_generator.manual_seed(seed)

    def init_device_mesh(self, tp_mesh):
        self.tp_mesh = tp_mesh

        # share noise between tp group
        group_seed = self.effective_rank
        self.setup_generator(seed=group_seed)

        # shard dit layers
        self.dit.init_device_mesh(tp_mesh)

    def get_l2_loss(self, model_output, target, w):
        return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)

    def forward(self, vid, text):
        device = vid.device

        alphas_cumprod_sqrt, idx = self.sigma_sampler(
            vid.shape[0], rand=None, return_idx=True, generator=self.noise_generator, device=device
        )
        noise = torch.randn(vid.shape, dtype=vid.dtype, device=vid.device, generator=self.noise_generator)

        noised_vid = vid.float() * append_dims(alphas_cumprod_sqrt, vid.ndim) + noise * append_dims(
            (1 - alphas_cumprod_sqrt**2) ** 0.5, vid.ndim  # type: ignore
        )

        sigma = append_dims(alphas_cumprod_sqrt, vid.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma, idx)

        model_output = self.dit(noised_vid * c_in, text, c_noise) * c_out + noised_vid * c_skip

        w = append_dims(1 / (1 - alphas_cumprod_sqrt**2), vid.ndim)  # type: ignore

        loss = self.get_l2_loss(model_output, vid, w)
        return loss
