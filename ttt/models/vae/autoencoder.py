import math
from typing import Any, Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.distributed

from ttt.models.vae.cp_enc_dec import ContextParallelDecoder3D, ContextParallelEncoder3D, _conv_gather, _conv_split
from ttt.models.vae.regularizers import DiagonalGaussianRegularizer
from ttt.models.vae.utils import (
    get_context_parallel_group,
    get_context_parallel_group_rank,
    initialize_context_parallel,
    is_context_parallel_initialized,
)


class AutoencodingEngine(pl.LightningModule):
    """
    Base class for all image autoencoders that we train, like VQGAN or AutoencoderKL
    (we also restore them explicitly as special cases for legacy reasons).
    Regularizations such as KL or VQ are moved to the regularizer class.
    """

    def __init__(
        self,
        encoder_config: Dict,
        decoder_config: Dict,
        *args,
        lr_g_factor: float = 1.0,
        disc_start_iter: int = 0,
        diff_boost_factor: float = 3.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False  # pytorch lightning

        self.encoder = ContextParallelEncoder3D(encoder_config)
        self.decoder = ContextParallelDecoder3D(decoder_config)
        self.regularization = DiagonalGaussianRegularizer()
        self.diff_boost_factor = diff_boost_factor
        self.disc_start_iter = disc_start_iter
        self.lr_g_factor = lr_g_factor

    def encode(
        self,
        x: torch.Tensor,
        return_reg_log: bool = False,
        unregularized: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        z = self.encoder(x, **kwargs)
        if unregularized:
            return z, dict()
        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.decoder(z, **kwargs)
        return x

    def forward(self, x: torch.Tensor, **additional_decode_kwargs) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        z, reg_log = self.encode(x, return_reg_log=True)
        dec = self.decode(z, **additional_decode_kwargs)
        return z, dec, reg_log


class VideoAutoencoderInferenceWrapper(AutoencodingEngine):
    def __init__(
        self,
        ckpt_path,
        encoder_config,
        decoder_config,
        scale_factor: float = 1.0,
        ignore_keys=list(),
    ):
        super().__init__(encoder_config, decoder_config)
        self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.cp_size = 1
        self.encoder_temporal_tiling_window = encoder_config.temporal_tiling_window
        self.decoder_temporal_tiling_window = decoder_config.temporal_tiling_window
        self.scale_factor = scale_factor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print("Missing keys: ", missing_keys)
        print("Unexpected keys: ", unexpected_keys)
        print(f"Restored from {path}")

    def encode(
        self,
        x: torch.Tensor,
        return_reg_log: bool = False,
        unregularized: bool = False,
        input_cp: bool = False,
        output_cp: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        x = x.contiguous()
        if self.cp_size > 0 and not input_cp:
            if not is_context_parallel_initialized():
                initialize_context_parallel(self.cp_size)

            global_src_rank = get_context_parallel_group_rank() * self.cp_size
            torch.distributed.broadcast(x, src=global_src_rank, group=get_context_parallel_group())

            x = _conv_split(x, dim=2, kernel_size=1)

        # if return_reg_log:
        #     z, reg_log = super().encode(x, return_reg_log, unregularized, **kwargs)
        # else:

        z = super().encode(x, return_reg_log, unregularized, **kwargs)

        if self.cp_size > 0 and not output_cp:
            z = _conv_gather(z, dim=2, kernel_size=1)

        # if return_reg_log:
        #     return z, reg_log
        return z

    def decode(
        self,
        z: torch.Tensor,
        input_cp: bool = False,
        output_cp: bool = False,
        split_kernel_size: int = 1,
        **kwargs,
    ):
        z = z.contiguous()
        if self.cp_size > 0 and not input_cp:
            if not is_context_parallel_initialized():
                initialize_context_parallel(self.cp_size)

            global_src_rank = get_context_parallel_group_rank() * self.cp_size
            torch.distributed.broadcast(z, src=global_src_rank, group=get_context_parallel_group())

            z = _conv_split(z, dim=2, kernel_size=split_kernel_size)

        x = super().decode(z, **kwargs)

        if self.cp_size > 0 and not output_cp:
            x = _conv_gather(x, dim=2, kernel_size=split_kernel_size)

        return x

    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     input_cp: bool = False,
    #     latent_cp: bool = False,
    #     output_cp: bool = False,
    #     **additional_decode_kwargs,
    # ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    #     z, reg_log = self.encode(x, return_reg_log=True, input_cp=input_cp, output_cp=latent_cp)
    #     dec = self.decode(z, input_cp=latent_cp, output_cp=output_cp, **additional_decode_kwargs)
    #     return z, dec, reg_log

    def forward(self, x) -> Any:
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.encode_first_stage(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        return x

    @torch.no_grad()
    def decode_first_stage(self, z: torch.Tensor) -> torch.Tensor:
        z = 1.0 / self.scale_factor * z
        n_samples = z.shape[0]
        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=False):
            for n in range(n_rounds):
                kwargs = {}
                if self.decoder_temporal_tiling_window:
                    out = []
                    window = self.decoder_temporal_tiling_window
                    for i in range(z.shape[2] // window):
                        start_frame, end_frame = (0, window + 1) if i == 0 else (window * i + 1, window * (i + 1) + 1)
                        z_part = z[n * n_samples : (n + 1) * n_samples, :, start_frame:end_frame].contiguous()
                        window_out = self.decode(z_part, clear_fake_cp_cache=(i + 1 == z.shape[2] // window), **kwargs)
                        out.append(window_out)
                    out = torch.cat(out, dim=2)
                else:
                    out = self.decode(z[n * n_samples : (n + 1) * n_samples], **kwargs)
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x, unregularized: bool = False, multiply_by_scale_factor: bool = False):
        n_samples = x.shape[0]
        n_rounds = math.ceil(x.shape[0] / n_samples)

        n_frames = x.shape[2]
        window = self.encoder_temporal_tiling_window
        assert window == 48
        n_windows = n_frames // window if n_frames > 1 else 1

        all_out = []
        with torch.autocast("cuda", enabled=False):
            for n in range(n_rounds):
                if self.encoder_temporal_tiling_window:
                    out = []
                    for i in range(n_windows):
                        start_frame, end_frame = (0, window + 1) if i == 0 else (window * i + 1, window * (i + 1) + 1)
                        x_part = x[n * n_samples : (n + 1) * n_samples, :, start_frame:end_frame].contiguous()
                        window_out = self.encode(
                            x_part, clear_fake_cp_cache=(i + 1 == n_windows), unregularized=unregularized
                        )
                        if unregularized:
                            assert isinstance(window_out, tuple)
                            window_out = window_out[0]
                        out.append(window_out)
                    out = torch.cat(out, dim=2)
                else:
                    if unregularized:
                        out, _ = self.encode(x[n * n_samples : (n + 1) * n_samples], unregularized=unregularized)
                    else:
                        out = self.encode(x[n * n_samples : (n + 1) * n_samples], unregularized=unregularized)
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        if multiply_by_scale_factor:
            z = self.scale_factor * z
        return z
