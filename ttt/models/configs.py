import json
from dataclasses import asdict, dataclass
from typing import List

from ttt.infra.config_manager import JobConfig


@dataclass
class ModelConfig:
    model_dim: int
    num_heads: int
    num_layers: int

    ssm_layer: str = "ttt_mlp"
    layer_norm_eps: float = 1e-6

    # TTT-Specific Configs
    mini_batch_size: int = 64
    ttt_base_lr: float = 0.1

    rope_theta: float = 10000
    scan_checkpoint_group_size: int = 16

    adapter_method: str = "none"  # none, sft, qkvo

    # Network Config
    time_embed_dim: int = 512
    sigma_interval: int = 1000
    patch_size: int = 2
    in_channels: int = 16
    out_channels: int = 16
    scale_factor: float = 1.0

    # ROPE Config
    latent_height: int = 30
    latent_width: int = 45
    compressed_num_frames: int = 13
    theta: float = 10000

    # Conditioner Config
    text_dim: int = 512

    # SSM Attn Config
    gating_alpha_init: float = 0.1
    attn_length: int = 12
    prefix_temporal_length: int = 1

    # Remat config
    remat_transformer_layer_group_size: int = 1
    remat_forward_ssm: bool = False
    remat_reverse_ssm: bool = False
    remat_attention: bool = False
    remat_mlp: bool = False
    remat_seq_modeling_block: bool = False
    shard_transformer_inputs: bool = False

    PREDEFINED_CONFIGS = {
        "debug": {
            "model_dim": 512,
            "num_heads": 8,
            "num_layers": 6,
        },
        "5B": {
            "model_dim": 3072,
            "num_heads": 48,
            "num_layers": 42,
            "text_dim": 4096,
        },
    }

    VIDEO_DURATION_CONFIGS = {
        "3sec": {
            "compressed_num_frames": 13,
        },
        "9sec": {
            "compressed_num_frames": 37,
        },
        "18sec": {
            "compressed_num_frames": 73,
        },
        "30sec": {
            "compressed_num_frames": 121,
        },
        "63sec": {
            "compressed_num_frames": 253,
        },
    }

    @classmethod
    def get_preset(cls, preset: str, video_length: str, job_config: JobConfig | None = None):
        if preset not in cls.PREDEFINED_CONFIGS:
            raise ValueError("Pre-defined config not found.")
        if video_length not in cls.VIDEO_DURATION_CONFIGS:
            raise ValueError("Pre-defined video duration config not found.")

        model_config = cls(**cls.PREDEFINED_CONFIGS[preset], **cls.VIDEO_DURATION_CONFIGS[video_length])  # type: ignore

        if job_config is not None:
            model_config.update(job_config)

        return model_config

    def __str__(self):
        return json.dumps(asdict(self), indent=4)

    def update(self, job_config: JobConfig):
        if job_config.training.adapter_method is not None:
            self.adapter_method = job_config.training.adapter_method

        self.scale_factor = job_config.model.scale_factor

        self.remat_transformer_layer_group_size = job_config.remat.transformer_checkpoint_layer_group_size
        self.remat_forward_ssm = job_config.remat.forward_ssm
        self.remat_reverse_ssm = job_config.remat.reverse_ssm
        self.remat_attention = job_config.remat.attention
        self.remat_mlp = job_config.remat.mlp
        self.remat_seq_modeling_block = job_config.remat.seq_modeling_block

        self.shard_transformer_inputs = job_config.remat.shard_transformer_inputs

        self.ssm_layer = job_config.model.ssm_layer

        self.mini_batch_size = job_config.model.mini_batch_size
        self.ttt_base_lr = job_config.model.ttt_base_lr
        self.scan_checkpoint_group_size = job_config.remat.scan_checkpoint_group_size


@dataclass
class VaeModelConfig:
    double_z: bool = True
    z_channels: int = 16
    resolution: int = 256
    in_channels: int = 3
    out_ch: int = 3
    ch: int = 128
    ch_mult: List[int] = (1, 2, 2, 4)  # type: ignore - cogvideo utils expect a tuple
    attn_resolutions: List[int] = ()  # type: ignore - cogvideo utils expect a tuple
    num_res_blocks: int = 3
    dropout: float = 0.0
    gather_norm: bool = True
    temporal_tiling_window: int = 16
    use_silu: bool = False

    @classmethod
    def get_encoder_config(cls, version=1.0, temporal_tiling_window=16):
        if version == 1.0:
            return cls(temporal_tiling_window=temporal_tiling_window)  # Return an instance with default values
        elif version == 1.5:
            return cls(use_silu=True, temporal_tiling_window=temporal_tiling_window)
        else:
            raise ValueError("ver1.0 or ver1.5 supported")

    @classmethod
    def get_decoder_config(cls, version=1.0, temporal_tiling_window=2):
        if version == 1.0:
            return cls(gather_norm=False, temporal_tiling_window=temporal_tiling_window)
        elif version == 1.5:
            return cls(gather_norm=False, use_silu=True, temporal_tiling_window=temporal_tiling_window)
        else:
            raise ValueError("ver1.0 or ver1.5 supported")
