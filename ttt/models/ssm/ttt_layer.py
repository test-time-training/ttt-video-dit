import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor

from ttt.models.cogvideo.utils import (SequenceMetadata, full_tensor,
                                       place_into, shard_tensor, to_local)
from ttt.models.configs import ModelConfig
from ttt.models.ssm.linear_triton import TritonLinear
from ttt.models.ssm.mlp_tk import TkMLP
from ttt.models.ssm.ops import ttt_linear, ttt_mlp
from ttt.models.ssm.utils import apply_rotary_emb, precompute_freqs_cis_3d


class TTTWrapper(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.model_dim = config.model_dim
        self.num_heads = config.num_heads
        self.rope_theta = config.rope_theta
        self.latent_height = config.latent_height
        self.latent_width = config.latent_width
        self.compressed_num_frames = config.compressed_num_frames

        if config.ssm_layer == "ttt_linear":
            self.ttt = TTTLinear(config)
        elif config.ssm_layer == "ttt_mlp":
            self.ttt = TTTMLP(config)
        else:
            raise TypeError(f"No ttt layer of type {config.ssm_layer}")

        self.register_buffer("freqs_cis", self._precompute_freqs_cis_3d(), persistent=False)

    def _precompute_freqs_cis_3d(self) -> torch.Tensor:
        return precompute_freqs_cis_3d(
            self.model_dim // self.num_heads,
            self.latent_height,
            self.latent_width,
            self.compressed_num_frames,
            self.rope_theta,
        )

    def init_freqs(self):
        self.freqs_cis.copy_(self._precompute_freqs_cis_3d())

    def forward(self, x: torch.Tensor, seq_metadata: SequenceMetadata):
        return self.ttt(x, self.freqs_cis, seq_metadata)


class TTTBase(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.width = config.model_dim
        self.num_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads
        self.mini_batch_size = config.mini_batch_size

        self.ttt_base_lr = config.ttt_base_lr
        self.scan_checkpoint_group_size = config.scan_checkpoint_group_size

        self.tp_mesh: None | DeviceMesh = None

        self._init_qkvo_proj()
        self._init_ttt_lr_gate()
        self._init_ttt_ln()

        self.post_norm = nn.LayerNorm(self.width, eps=1e-6)

    # We must reinitialize after meta initialization
    def init_weights(self):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.wo.weight, mean=0.0, std=0.02)

        self.post_norm.reset_parameters()
        nn.init.ones_(self.ttt_norm_weight.data)
        nn.init.zeros_(self.ttt_norm_bias)
        nn.init.normal_(self.learnable_ttt_lr_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.learnable_ttt_lr_bias)

    def _init_qkvo_proj(self):
        self.wq = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)
        self.wk = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)
        self.wv = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)
        self.wo = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)

    def _init_ttt_lr_gate(self):
        linear_weight_data = nn.Linear(self.width, 1, bias=True).weight.data
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.stack(
                [torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)],
                dim=0,
            )
        )

        linear_bias_data = nn.Linear(self.width, 1, bias=True).bias.data
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.stack(
                [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
                dim=0,
            )
        )

    def _init_ttt_ln(self):
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))

    def init_device_mesh(self, tp_mesh: DeviceMesh):
        self.tp_mesh = tp_mesh

        self.ttt_norm_weight = nn.Parameter(distribute_tensor(self.ttt_norm_weight, tp_mesh, [Shard(0)]))
        self.ttt_norm_bias = nn.Parameter(distribute_tensor(self.ttt_norm_bias, tp_mesh, [Shard(0)]))

        self.learnable_ttt_lr_weight = nn.Parameter(
            distribute_tensor(self.learnable_ttt_lr_weight, tp_mesh, [Replicate()])
        )
        self.learnable_ttt_lr_bias = nn.Parameter(distribute_tensor(self.learnable_ttt_lr_bias, tp_mesh, [Replicate()]))

    def shard_inputs(self, inputs):
        assert self.tp_mesh is not None, "Tensor parallel mesh must be initialized before sharding inputs."

        for key in inputs:
            assert inputs[key].shape[1] == self.num_heads, "Sharding is only supported on the head dimension."
            inputs[key] = shard_tensor(inputs[key], self.tp_mesh, dim=1)

        return inputs

    @torch.compile
    def get_qkv_projections(self, hidden_states):
        XQ, XK, XV = (
            self.wq(hidden_states),
            self.wk(hidden_states),
            self.wv(hidden_states),
        )
        return XQ, XK, XV

    @torch.compile
    def get_eta(self, X):
        learnable_ttt_lr_weight = full_tensor(self.learnable_ttt_lr_weight)
        learnable_ttt_lr_bias = full_tensor(self.learnable_ttt_lr_bias)

        ttt_lr = torch.einsum("bnkc,hdc->bhnkd", X, learnable_ttt_lr_weight) + learnable_ttt_lr_bias.reshape(
            1, -1, 1, 1, 1
        )  # [B,nc,cs,c] @ [nh,1,c] -> [B,nh,nc,cs,1] + [1,nh,1,1,1] -> [B,nh,nc,cs,1]

        ttt_lr = F.sigmoid(ttt_lr)  # [B,H,nc,K,1]

        ttt_lr = ttt_lr.permute(0, 1, 2, 4, 3)
        return self.ttt_base_lr * ttt_lr / self.head_dim

    @torch.compile
    def interleave(self, x: torch.Tensor, seq_metadata: SequenceMetadata):
        init_offset, num_chunks, text_length = (
            seq_metadata.init_offset,
            seq_metadata.num_chunks,
            seq_metadata.text_length,
        )
        assert init_offset is not None, "Init offset must be provided for interleaving."

        seq_text_length = text_length * num_chunks

        B, H, NC, C, HD = x.shape
        x_flatten = x.reshape(B, H, NC * C, HD)

        x_text = x_flatten[:, :, :seq_text_length]
        x_video = x_flatten[:, :, seq_text_length:]

        # Get individual scene text embeddings.
        x_text = torch.chunk(x_text, num_chunks, dim=2)

        # The first scene will have one extra latent frame.
        video_init_offset = init_offset - text_length
        partial_chunks = torch.chunk(x_video[:, :, video_init_offset:], num_chunks - 1, dim=2)
        x_video = (x_video[:, :, :video_init_offset],) + partial_chunks

        x_interleaved = []
        for i in range(num_chunks):
            x_interleaved.append(torch.cat((x_text[i], x_video[i]), dim=2))

        return torch.cat(x_interleaved, dim=2).reshape(B, H, NC, C, HD)

    @torch.compile
    def undo_interleave(self, x: torch.Tensor, seq_metadata: SequenceMetadata):
        text_length, init_offset, base_offset, num_chunks = (
            seq_metadata.text_length,
            seq_metadata.init_offset,
            seq_metadata.base_offset,
            seq_metadata.num_chunks,
        )

        assert base_offset is not None, "Base offset must be provided for undoing interleaving."
        assert init_offset is not None, "Init offset must be provided for undoing interleaving."

        text_embs, vid_embs = torch.tensor([], dtype=x.dtype, device=x.device), torch.tensor(
            [], dtype=x.dtype, device=x.device
        )

        for i in range(num_chunks):
            if i == 0:
                scene_start_idx = 0
                scene_end_idx = init_offset
            else:
                scene_start_idx = init_offset + (i - 1) * base_offset
                scene_end_idx = init_offset + i * base_offset

            scene_emb = x[:, scene_start_idx:scene_end_idx]

            text_embs = torch.cat((text_embs, scene_emb[:, :text_length]), dim=1)
            vid_embs = torch.cat((vid_embs, scene_emb[:, text_length:]), dim=1)

        return torch.cat((text_embs, vid_embs), dim=1)

    @torch.compile
    def ln_reconstruction_target(self, XV, XK):
        XV = XV - XK
        eps = 1e-8
        # Compute mean and std over the head dimension (last dimension)
        mean = XV.mean(dim=-1, keepdim=True)
        std = place_into(to_local(XV).std(dim=-1, keepdim=True), XV)

        # Normalize
        XV = (XV - mean) / (std + eps)

        # Apply per-head weight and bias.
        # self.ttt_norm_weight and self.ttt_norm_bias have shape [num_heads, head_dim].
        # We unsqueeze to make them broadcastable with XV_norm which is [B, L, num_heads, head_dim].
        XV = self.ttt_norm_weight.unsqueeze(0).unsqueeze(0) * XV + self.ttt_norm_bias.unsqueeze(0).unsqueeze(0)

        return XV + XK

    @torch.compile
    def reshape_to_mini_batch(self, X, XQ, XK, XV):
        B, L = X.shape[:2]
        num_mini_batch = L // self.mini_batch_size

        XQ, XK, XV = XQ.transpose(1, 2), XK.transpose(1, 2), XV.transpose(1, 2)

        X = X.reshape(B, num_mini_batch, self.mini_batch_size, self.width)

        XQ = XQ.reshape(B, self.num_heads, num_mini_batch, self.mini_batch_size, self.head_dim)
        XK = XK.reshape(B, self.num_heads, num_mini_batch, self.mini_batch_size, self.head_dim)
        XV = XV.reshape(B, self.num_heads, num_mini_batch, self.mini_batch_size, self.head_dim)

        return X, XQ, XK, XV

    def process_input(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor, seq_metadata: SequenceMetadata):
        seq_text_length = seq_metadata.seq_text_length

        B, L = hidden_states.shape[:2]
        mini_batch_size = self.mini_batch_size

        XQ, XK, XV = self.get_qkv_projections(hidden_states)

        XQ = XQ.view(B, L, -1, self.head_dim)
        XK = XK.view(B, L, -1, self.head_dim)
        XV = XV.view(B, L, -1, self.head_dim)

        # L2 Norm
        XQ = place_into(torch.nn.functional.normalize(to_local(XQ), p=2, dim=-1), XQ)
        XK = place_into(torch.nn.functional.normalize(to_local(XK), p=2, dim=-1), XK)

        XQ_text, XQ_video = XQ[:, :seq_text_length], XQ[:, seq_text_length:]
        XK_text, XK_video = XK[:, :seq_text_length], XK[:, seq_text_length:]

        XQ_rope_video, XK_rope_video = apply_rotary_emb(
            to_local(XQ_video), to_local(XK_video), freqs_cis=to_local(freqs_cis)
        )

        XQ_video = place_into(XQ_rope_video, XQ_video)
        XK_video = place_into(XK_rope_video, XK_video)

        XQ = torch.cat((XQ_text, XQ_video), dim=1)
        XK = torch.cat((XK_text, XK_video), dim=1)

        XV = self.ln_reconstruction_target(XV, XK)

        hidden_states, XQ, XK, XV = self.reshape_to_mini_batch(hidden_states, XQ, XK, XV)

        ttt_lr_eta = self.get_eta(hidden_states)

        # We do not use token_eta for non-causal chunks
        eta = 1 / mini_batch_size * ttt_lr_eta.repeat(1, 1, 1, mini_batch_size, 1)

        if seq_metadata.is_multiscene:
            XQ = place_into(self.interleave(to_local(XQ), seq_metadata), XQ)
            XK = place_into(self.interleave(to_local(XK), seq_metadata), XK)
            XV = place_into(self.interleave(to_local(XV), seq_metadata), XV)
            eta = self.interleave(to_local(eta), seq_metadata)

        inputs = {
            "XQ": XQ,
            "XK": XK,
            "XV": XV,
            "eta": eta,
        }

        if self.tp_mesh is not None:
            inputs = self.shard_inputs(inputs)

        return inputs

    def ttt(
        self,
        inputs,
    ):
        raise NotImplementedError("ttt method must be implemented in TTTBase subclasses.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        seq_metadata: SequenceMetadata,
    ):
        assert (
            hidden_states.size(1) % self.config.mini_batch_size == 0
        ), "Sequence len must be multiple of mini batch size."

        hidden_states = self.ttt(self.process_input(hidden_states, freqs_cis, seq_metadata))

        hidden_states = self.post_norm(hidden_states)
        hidden_states = self.wo(hidden_states)

        hidden_states = full_tensor(hidden_states)

        if seq_metadata.is_multiscene:
            hidden_states = self.undo_interleave(to_local(hidden_states), seq_metadata)

        return hidden_states


class TTTLinear(TTTBase):
    def __init__(self, config: ModelConfig, use_kernel: bool = True):
        super().__init__(config)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

        # For acceleration
        self.use_kernel = use_kernel

    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.W1, mean=0.0, std=0.02)
        nn.init.zeros_(self.b1)

    def init_device_mesh(self, tp_mesh: DeviceMesh):
        assert self.use_kernel, "Tensor parallel is not currently supported for TTTLinear without kernel."
        super().init_device_mesh(tp_mesh)

        self.W1 = nn.Parameter(distribute_tensor(self.W1, tp_mesh, [Shard(0)]))
        self.b1 = nn.Parameter(distribute_tensor(self.b1, tp_mesh, [Shard(0)]))

        TritonLinear.sharded_mode = True

    def ttt(self, inputs):
        B = inputs["XV"].shape[0]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        num_mini_batch = inputs["XV"].shape[2]

        W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
        b1_states = torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1))

        checkpoint_group_size = min(max(self.config.scan_checkpoint_group_size, 1), num_mini_batch)

        if self.use_kernel:
            XQW_batch = TritonLinear.apply(
                self.ttt_norm_weight,
                self.ttt_norm_bias,
                W1_states,
                b1_states,
                inputs["XQ"],
                inputs["XV"],
                inputs["XK"],
                inputs["eta"],
                checkpoint_group_size,
            )

            XQW_batch = XQW_batch.permute(0, 2, 3, 1, 4)
        else:
            XQW_batch = ttt_linear(
                inputs["XK"],
                inputs["XQ"],
                inputs["XV"],
                inputs["eta"],
                self.ttt_norm_weight,
                self.ttt_norm_bias,
                W1_states,
                b1_states,
                checkpoint_group_size,
            )

        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch


class TTTMLP(TTTBase):
    def __init__(self, config: ModelConfig, use_kernel: bool = True):
        super().__init__(config)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

        self.use_kernel = use_kernel

    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.W1, mean=0.0, std=0.02)
        nn.init.zeros_(self.b1)
        nn.init.normal_(self.W2, mean=0.0, std=0.02)
        nn.init.zeros_(self.b2)

    def init_device_mesh(self, tp_mesh: DeviceMesh):
        assert self.use_kernel, "Tensor parallel is not currently supported for TTTMLP without kernel."
        super().init_device_mesh(tp_mesh)

        self.W1 = nn.Parameter(distribute_tensor(self.W1, tp_mesh, [Shard(0)]))
        self.b1 = nn.Parameter(distribute_tensor(self.b1, tp_mesh, [Shard(0)]))
        self.W2 = nn.Parameter(distribute_tensor(self.W2, tp_mesh, [Shard(0)]))
        self.b2 = nn.Parameter(distribute_tensor(self.b2, tp_mesh, [Shard(0)]))

        TkMLP.sharded_mode = True

    def ttt(self, inputs):
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]

        W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
        b1_states = torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1))
        W2_states = torch.tile(self.W2.unsqueeze(0), dims=(B, 1, 1, 1))
        b2_states = torch.tile(self.b2.unsqueeze(0), dims=(B, 1, 1, 1))

        checkpoint_group_size = min(max(self.config.scan_checkpoint_group_size, 1), num_mini_batch)

        if self.use_kernel:
            XQW_batch = TkMLP.apply(
                self.ttt_norm_weight,
                self.ttt_norm_bias,
                W1_states,
                b1_states,
                W2_states,
                b2_states,
                inputs["XQ"],
                inputs["XV"],
                inputs["XK"],
                inputs["eta"],
                checkpoint_group_size,
            )

            XQW_batch = XQW_batch.permute(0, 2, 3, 1, 4)
        else:
            XQW_batch = ttt_mlp(
                inputs["XK"],
                inputs["XQ"],
                inputs["XV"],
                inputs["eta"],
                self.ttt_norm_weight,
                self.ttt_norm_bias,
                W1_states,
                b1_states,
                W2_states,
                b2_states,
                checkpoint_group_size,
            )

        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch
