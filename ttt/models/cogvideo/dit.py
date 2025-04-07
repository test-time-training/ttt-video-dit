from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, distribute_tensor

from ttt.models.cogvideo.utils import (Rotary3DPositionEmbedding,
                                       SequenceMetadata, full_tensor, modulate,
                                       shard_tensor, timestep_embedding,
                                       unpatchify)
from ttt.models.configs import ModelConfig


class PatchEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        requires_grad = config.adapter_method == "sft"

        self.vid_proj = nn.Conv2d(
            config.in_channels,
            config.model_dim,
            config.patch_size,
            config.patch_size,
            bias=True,
        ).requires_grad_(requires_grad)
        self.text_proj = nn.Linear(config.text_dim, config.model_dim, bias=True).requires_grad_(requires_grad)

    def forward(self, video, text_encoding):
        batch_size, num_frames = video.shape[:2]

        vid_emb = rearrange(video, "b t c h w -> (b t) c h w")
        vid_emb = self.vid_proj(vid_emb)
        vid_emb = rearrange(vid_emb, "(b t) c h w -> b (t h w) c", b=batch_size, t=num_frames).contiguous()

        text_emb = self.text_proj(text_encoding).contiguous()

        return text_emb, vid_emb


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        requires_grad = config.adapter_method == "sft"

        self.do_remat = config.remat_mlp
        self.requires_grad = requires_grad

        self.layer1 = nn.Linear(config.model_dim, 4 * config.model_dim, bias=True).requires_grad_(requires_grad)
        self.layer2 = nn.Linear(4 * config.model_dim, config.model_dim, bias=True).requires_grad_(requires_grad)

        self.tp_mesh: None | DeviceMesh = None

    def init_device_mesh(self, tp_mesh: DeviceMesh):
        self.tp_mesh = tp_mesh

        # Replicate params for sequence parallel
        self.layer1.weight = nn.Parameter(distribute_tensor(self.layer1.weight, tp_mesh, [Replicate()])).requires_grad_(
            self.requires_grad
        )
        self.layer1.bias = nn.Parameter(distribute_tensor(self.layer1.bias, tp_mesh, [Replicate()])).requires_grad_(
            self.requires_grad
        )

        self.layer2.weight = nn.Parameter(distribute_tensor(self.layer2.weight, tp_mesh, [Replicate()])).requires_grad_(
            self.requires_grad
        )
        self.layer2.bias = nn.Parameter(distribute_tensor(self.layer2.bias, tp_mesh, [Replicate()])).requires_grad_(
            self.requires_grad
        )

    def forward(self, x):

        @torch.compile
        def mlp_forward(x):
            x = self.layer1(x)
            x = F.gelu(x, approximate="tanh")
            x = self.layer2(x)
            return full_tensor(x)

        if self.do_remat:
            mlp_forward = partial(torch.utils.checkpoint.checkpoint, mlp_forward, use_reentrant=False)

        x = mlp_forward(x)
        return x


class SSMGating(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.gating_alpha = nn.Parameter(torch.ones(config.model_dim) * config.gating_alpha_init)

    def forward(self, x):
        gating_alpha = full_tensor(self.gating_alpha)

        gating_alpha = torch.tanh(gating_alpha)
        return gating_alpha * x


class SeqModelingBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        requires_grad = config.adapter_method == "sft" or config.adapter_method == "qkvo"

        self.do_attn_remat = config.remat_attention
        self.do_forward_ssm_remat = config.remat_forward_ssm
        self.do_reverse_ssm_remat = config.remat_reverse_ssm

        self.num_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads
        self.prefix_temporal_length = config.prefix_temporal_length
        self.attn_length = config.attn_length

        self.q_norm = nn.LayerNorm(self.head_dim, eps=config.layer_norm_eps)
        for param in self.q_norm.parameters():
            param.requires_grad = requires_grad

        self.k_norm = nn.LayerNorm(self.head_dim, eps=config.layer_norm_eps)
        for param in self.k_norm.parameters():
            param.requires_grad = requires_grad

        self.rotary = Rotary3DPositionEmbedding(
            config.latent_height,
            config.latent_width,
            config.compressed_num_frames,
            self.head_dim,
            config.theta,
        )

        assert config.adapter_method in ("sft", "qkvo", "none"), f"Invalid adapter method: {config.adapter_method}"

        self.q = nn.Linear(config.model_dim, config.model_dim, bias=True)
        self.k = nn.Linear(config.model_dim, config.model_dim, bias=True)
        self.v = nn.Linear(config.model_dim, config.model_dim, bias=True)
        self.o = nn.Linear(config.model_dim, config.model_dim, bias=True)

        from ttt.models.ssm.ttt_layer import TTTWrapper

        self.ssm = TTTWrapper(config)

        self.forward_ssm_gating_video = SSMGating(config)
        self.forward_ssm_gating_text = SSMGating(config)
        self.backward_ssm_gating_video = SSMGating(config)
        self.backward_ssm_gating_text = SSMGating(config)

    def init_device_mesh(self, tp_mesh: DeviceMesh):
        def replicate_gate(gate: SSMGating):
            gate.gating_alpha = nn.Parameter(distribute_tensor(gate.gating_alpha, tp_mesh, [Replicate()]))

        replicate_gate(self.forward_ssm_gating_text)
        replicate_gate(self.forward_ssm_gating_video)
        replicate_gate(self.backward_ssm_gating_text)
        replicate_gate(self.backward_ssm_gating_video)

        self.rotary.init_device_mesh(tp_mesh)

    def _attn_forward(self, vid_emb: torch.Tensor, text_emb: torch.Tensor, seq_metadata: SequenceMetadata):
        text_length, tokens_per_frame = (seq_metadata.text_length, seq_metadata.tokens_per_frame)

        num_attn_steps = seq_metadata.num_chunks
        output_vid_emb, output_text_emb = torch.zeros_like(vid_emb), torch.zeros_like(text_emb)

        output_vid_overlap_count = torch.zeros_like(vid_emb[..., 0:1])

        for i in range(num_attn_steps):
            start_idx = i * self.attn_length * tokens_per_frame
            end_idx = (self.prefix_temporal_length + (i + 1) * self.attn_length) * tokens_per_frame

            start_txt_idx = i * text_length
            end_txt_idx = (i + 1) * text_length

            target_text_emb = text_emb[:, start_txt_idx:end_txt_idx]

            cur_emb = torch.cat([target_text_emb, vid_emb[:, start_idx:end_idx]], dim=1)

            @torch.compile
            def do_attention(cur_emb):
                cur_q = rearrange(self.q(cur_emb), "b t (h d) -> b h t d", d=self.head_dim)
                cur_k = rearrange(self.k(cur_emb), "b t (h d) -> b h t d", d=self.head_dim)
                cur_v = rearrange(self.v(cur_emb), "b t (h d) -> b h t d", d=self.head_dim)

                cur_q = self.q_norm(cur_q)
                cur_k = self.k_norm(cur_k)

                new_k_emb = self.rotary(cur_k[:, :, text_length:])
                new_q_emb = self.rotary(cur_q[:, :, text_length:])
                cur_k = torch.cat([cur_k[:, :, :text_length], new_k_emb], dim=2)
                cur_q = torch.cat([cur_q[:, :, :text_length], new_q_emb], dim=2)

                cur_attn = F.scaled_dot_product_attention(
                    cur_q, cur_k, cur_v, attn_mask=None, dropout_p=0, is_causal=False
                )
                cur_attn = rearrange(cur_attn, "b h t d -> b t (h d)")
                return self.o(cur_attn)

            attn_output = do_attention(cur_emb)

            output_text_emb[:, start_txt_idx:end_txt_idx] = attn_output[:, :text_length]

            output_vid_emb[:, start_idx:end_idx] += attn_output[:, text_length:]
            output_vid_overlap_count[:, start_idx:end_idx] += 1

        output_vid_emb = output_vid_emb / output_vid_overlap_count

        return torch.cat((output_text_emb, output_vid_emb), dim=1)

    def _reverse_text_chunks(self, text_emb, num_chunks):
        original_text_emb_shape = text_emb.shape
        text_emb = rearrange(text_emb, "b (c s) e -> b c s e", c=num_chunks)
        text_emb = torch.flip(text_emb, dims=[1])
        return text_emb.view(original_text_emb_shape)

    def _gate(self, text_gate, video_gate, residual, ssm_output, text_length):
        return residual + torch.cat(
            [text_gate(ssm_output[:, :text_length]), video_gate(ssm_output[:, text_length:])], dim=1
        )

    def _ssm_forward(self, emb: torch.Tensor, seq_metadata: SequenceMetadata):
        text_length, num_chunks = seq_metadata.seq_text_length, seq_metadata.num_chunks

        # Apply remat if configured
        # Note: both forward and reverse ssm use the same ssm layer and parameters
        forward_ssm = (
            partial(torch.utils.checkpoint.checkpoint, self.ssm, use_reentrant=False)
            if self.do_forward_ssm_remat
            else self.ssm
        )
        reverse_ssm = (
            partial(torch.utils.checkpoint.checkpoint, self.ssm, use_reentrant=False)
            if self.do_reverse_ssm_remat
            else self.ssm
        )

        # Embedding pre-forward ssm
        residual_emb = emb.clone()

        emb = forward_ssm(emb, seq_metadata)

        emb = self._gate(self.forward_ssm_gating_text, self.forward_ssm_gating_video, residual_emb, emb, text_length)

        # Embedding pre-reversed ssm
        residual_emb = emb.clone()

        # Reverse the text chunks to match reversed video chunks
        if seq_metadata.is_multiscene:
            emb[:, :text_length] = self._reverse_text_chunks(emb[:, :text_length], num_chunks)

        # Reverse the video latent
        emb[:, text_length:] = torch.flip(residual_emb[:, text_length:], dims=[1])

        emb = reverse_ssm(emb, seq_metadata)

        # Unreverse the text chunks to match reversed video chunks
        if seq_metadata.is_multiscene:
            emb[:, :text_length] = self._reverse_text_chunks(emb[:, :text_length], num_chunks)

        # Unreverse the video latent
        emb[:, text_length:] = torch.flip(emb[:, text_length:], dims=[1])

        return self._gate(self.backward_ssm_gating_text, self.backward_ssm_gating_video, residual_emb, emb, text_length)

    def forward(self, vid_emb: torch.Tensor, text_emb: torch.Tensor, seq_metadata: SequenceMetadata):
        if self.do_attn_remat:
            output = torch.utils.checkpoint.checkpoint(
                self._attn_forward, vid_emb, text_emb, seq_metadata, use_reentrant=False
            )
        else:
            output = self._attn_forward(vid_emb, text_emb, seq_metadata)

        output = self._ssm_forward(output, seq_metadata)

        return output[:, seq_metadata.seq_text_length :], output[:, : seq_metadata.seq_text_length]


class TransformerLayer(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        requires_grad = config.adapter_method == "sft"

        self.remat_seq_modeling_block = config.remat_seq_modeling_block

        self.tp_mesh: None | DeviceMesh = None

        self.pre_seq_layernorm = nn.LayerNorm(config.model_dim, eps=config.layer_norm_eps)
        for param in self.pre_seq_layernorm.parameters():
            param.requires_grad = requires_grad
        self.pre_seq_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.time_embed_dim, 6 * config.model_dim, bias=True).requires_grad_(requires_grad),
        )

        self.seq_modeling_block = SeqModelingBlock(config)

        self.pre_mlp_layernorm = nn.LayerNorm(config.model_dim, eps=config.layer_norm_eps)
        for param in self.pre_mlp_layernorm.parameters():
            param.requires_grad = requires_grad
        self.pre_mlp_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.time_embed_dim, 6 * config.model_dim, bias=True).requires_grad_(requires_grad),
        )

        self.mlp = MLP(config)

    def init_device_mesh(self, tp_mesh: DeviceMesh):
        self.tp_mesh = tp_mesh

        # init module device meshes
        self.mlp.init_device_mesh(tp_mesh)
        self.seq_modeling_block.init_device_mesh(tp_mesh)
        self.seq_modeling_block.ssm.ttt.init_device_mesh(tp_mesh)

    def forward(self, vid_emb: torch.Tensor, text_emb: torch.Tensor, seq_metadata: SequenceMetadata):
        text_length = seq_metadata.seq_text_length
        seq_modeling_block = (
            partial(torch.utils.checkpoint.checkpoint, self.seq_modeling_block, use_reentrant=False)
            if self.remat_seq_modeling_block
            else self.seq_modeling_block
        )

        (
            shift_msa,
            scale_msa,
            gate_msa,
            text_shift_msa,
            text_scale_msa,
            text_gate_msa,
        ) = self.pre_seq_adaLN_modulation(seq_metadata.t_emb).chunk(6, dim=1)

        gate_msa, text_gate_msa = (
            gate_msa.unsqueeze(1),
            text_gate_msa.unsqueeze(1),
        )

        vid_seq_input = modulate(self.pre_seq_layernorm(vid_emb), shift_msa, scale_msa)
        text_seq_input = modulate(self.pre_seq_layernorm(text_emb), text_shift_msa, text_scale_msa)

        vid_seq_output, text_seq_output = seq_modeling_block(vid_seq_input, text_seq_input, seq_metadata)

        vid_emb = vid_emb + gate_msa * vid_seq_output
        text_emb = text_emb + text_gate_msa * text_seq_output

        (
            shift_mlp,
            scale_mlp,
            gate_mlp,
            text_shift_mlp,
            text_scale_mlp,
            text_gate_mlp,
        ) = self.pre_mlp_adaLN_modulation(seq_metadata.t_emb).chunk(6, dim=1)

        gate_mlp, text_gate_mlp = (
            gate_mlp.unsqueeze(1),
            text_gate_mlp.unsqueeze(1),
        )

        vid_mlp_input = modulate(self.pre_mlp_layernorm(vid_emb), shift_mlp, scale_mlp)
        text_mlp_input = modulate(self.pre_mlp_layernorm(text_emb), text_shift_mlp, text_scale_mlp)

        mlp_input = torch.cat((text_mlp_input, vid_mlp_input), dim=1)

        # Sequence parallel for mlp
        if self.tp_mesh is not None:
            mlp_input = shard_tensor(mlp_input, self.tp_mesh, dim=1)

        mlp_output = self.mlp(mlp_input)

        text_mlp_output = mlp_output[:, :text_length]
        vid_mlp_output = mlp_output[:, text_length:]

        vid_emb = vid_emb + gate_mlp * vid_mlp_output
        text_emb = text_emb + text_gate_mlp * text_mlp_output

        return vid_emb, text_emb


class FinalLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        requires_grad = config.adapter_method == "sft"

        self.out_channels = config.out_channels
        self.patch_size = config.patch_size

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.time_embed_dim, 2 * config.model_dim, bias=True).requires_grad_(requires_grad),
        )

        self.norm = nn.LayerNorm(config.model_dim, elementwise_affine=True, eps=config.layer_norm_eps)
        for param in self.norm.parameters():
            param.requires_grad = requires_grad
        self.linear = nn.Linear(
            config.model_dim,
            config.patch_size * config.patch_size * self.out_channels,
            bias=True,
        ).requires_grad_(requires_grad)

    def forward(self, vid_emb: torch.Tensor, seq_metadata: SequenceMetadata):
        shift, scale = self.adaLN_modulation(seq_metadata.t_emb).chunk(2, dim=1)
        vid_emb = modulate(self.norm(vid_emb), shift, scale)
        vid_emb = self.linear(vid_emb)

        return unpatchify(
            vid_emb,
            c=self.out_channels,
            p=self.patch_size,
            w=seq_metadata.latent_width // self.patch_size,
            h=seq_metadata.latent_height // self.patch_size,
        )


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        requires_grad = config.adapter_method == "sft"

        self.frames_per_chunk = config.attn_length
        self.remat_transformer_layer_group_size = config.remat_transformer_layer_group_size
        assert (
            config.num_layers % self.remat_transformer_layer_group_size == 0
        ), "Remat group size must be divisible into num layers"

        self.model_dim = config.model_dim
        self.shard_transformer_inputs = config.shard_transformer_inputs

        self.time_embed = nn.Sequential(
            nn.Linear(config.model_dim, config.time_embed_dim, bias=True).requires_grad_(requires_grad),
            nn.SiLU(),
            nn.Linear(config.time_embed_dim, config.time_embed_dim, bias=True).requires_grad_(requires_grad),
        )
        self.patch_embedding = PatchEmbedding(config)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)])

        self.transformer_norm = nn.LayerNorm(config.model_dim, eps=config.layer_norm_eps)
        for param in self.transformer_norm.parameters():
            param.requires_grad = requires_grad

        self.final_layer = FinalLayer(config)

    def init_device_mesh(self, tp_mesh: DeviceMesh):
        self.tp_mesh = tp_mesh

        for layer in self.layers:
            layer.init_device_mesh(tp_mesh)  # type: ignore

    def forward(self, video, text, timesteps):
        num_frames, height, width = video.shape[1], video.shape[3], video.shape[4]
        text_length = text.shape[-2]

        # Timestep embeddings
        t_emb = timestep_embedding(timesteps, self.model_dim, dtype=video.dtype)
        t_emb = self.time_embed(t_emb)

        # Image patch / text embeddings
        text_emb, vid_emb = self.patch_embedding(video, text)

        num_chunks = text_emb.shape[1]
        seq_metadata = SequenceMetadata(
            text_length=text_length,
            seq_text_length=text_length * num_chunks,
            num_frames=num_frames,
            num_chunks=num_chunks,
            tokens_per_frame=(vid_emb.shape[1]) // num_frames,
            latent_height=height,
            latent_width=width,
            t_emb=t_emb,
        )

        # Get offsets to handle interleaving
        if seq_metadata.is_multiscene:
            seq_metadata.init_multiscene_offsets()

        text_emb = rearrange(text_emb, "b c s e -> b (c s) e")

        def checkpointed_group_forward(
            i: int, vid_emb: torch.Tensor, text_emb: torch.Tensor, seq_metadata: SequenceMetadata
        ):
            for layer in self.layers[i : i + self.remat_transformer_layer_group_size]:
                vid_emb, text_emb = layer(full_tensor(vid_emb), full_tensor(text_emb), seq_metadata)
            return vid_emb, text_emb

        for i in range(0, len(self.layers), self.remat_transformer_layer_group_size):
            if self.shard_transformer_inputs:
                assert self.tp_mesh is not None, "Sharding requires tensor parallel mesh to be set"
                vid_emb = shard_tensor(vid_emb, self.tp_mesh, dim=1)
                text_emb = shard_tensor(text_emb, self.tp_mesh, dim=1)

            vid_emb, text_emb = torch.utils.checkpoint.checkpoint(
                checkpointed_group_forward, i, vid_emb, text_emb, seq_metadata, use_reentrant=False
            )

        vid_emb = self.transformer_norm(vid_emb)
        return self.final_layer(vid_emb, seq_metadata)
