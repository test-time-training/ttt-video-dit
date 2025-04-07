import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from transformers import T5EncoderModel, T5Tokenizer

from ttt.datasets.preembedding_dataset import SCENE_END_TOKEN, SCENE_START_TOKEN
from ttt.infra.checkpoint import MODEL_STATE_DICT_KEY
from ttt.infra.config_manager import JobConfig
from ttt.infra.parallelisms import TORCH_DTYPE_MAP, apply_parallelisms
from ttt.models.cogvideo.model import CogVideoX
from ttt.models.cogvideo.utils import DiscreteDenoiser, VPSDEDPMPP2MSampler, cast_rotary_freqs
from ttt.models.configs import ModelConfig, VaeModelConfig
from ttt.models.vae.autoencoder import VideoAutoencoderInferenceWrapper


@dataclass(frozen=False)
class SceneDescription:
    """Dataclass to hold prompt information."""

    text: str
    requires_scene_transition: bool = False
    neg_text: Optional[str] = None


class PromptManager:
    """Handle loading and processing of text prompts for video generation."""

    @staticmethod
    def get_prompts(input_file: str) -> List[Tuple[List[str], Optional[List[str]]]]:
        prompts, neg_prompts = PromptManager._parse_jsonl_prompts_from_file(input_file)

        return list(zip(prompts, neg_prompts))

    @staticmethod
    def _parse_jsonl_prompts_from_file(path: str) -> tuple[List[List[str]], List[List[str]]]:
        """Parse prompts from a JSONL file."""

        is_jsonl = str(path).endswith(".jsonl")
        if not is_jsonl:
            assert str(path).endswith(".json"), "Invalid prompt file format. Expected .jsonl or .json"

        def _insert_special_tokens(scenes: List[SceneDescription]):
            for i, scene in enumerate(scenes):
                if i == 0:
                    # remove scene start token from first scene
                    scene.requires_scene_transition = False

                if scene.requires_scene_transition:
                    # add scene end token to previous scene and scene start token to current scene
                    scenes[i - 1].text += SCENE_END_TOKEN
                    scenes[i].text = SCENE_START_TOKEN + scenes[i].text

        text, neg_text = [], []

        with open(path, "r", encoding="utf-8") as f:
            if is_jsonl:
                video_descriptions = [json.loads(line) for line in f.readlines()]
            else:
                video_descriptions = json.load(f)

            for video_description in video_descriptions:
                scenes: List[SceneDescription] = [SceneDescription(**obj) for obj in video_description]

                _insert_special_tokens(scenes)
                text.append([scene.text for scene in scenes])
                neg_text.append([scene.neg_text for scene in scenes])

        return text, neg_text


class ModelLoader:
    """Handle loading and initialization of models for inference."""

    @staticmethod
    def load_t5_encoder(job_config: JobConfig, device: str) -> Tuple[T5Tokenizer, T5EncoderModel]:
        """
        Load and initialize the T5 tokenizer and encoder model.

        Returns:
            Tuple of (tokenizer, encoder)
        """
        t5_dir = job_config.eval.t5_model_dir
        assert t5_dir is not None, "T5 model directory not provided"

        tokenizer = T5Tokenizer.from_pretrained(t5_dir)
        encoder = T5EncoderModel.from_pretrained(t5_dir, torch_dtype=TORCH_DTYPE_MAP[job_config.eval.dtype])

        print(f"Adding scene tokens to tokenizer... {encoder.config.vocab_size}", end=" -> ")
        tokenizer.add_special_tokens({"additional_special_tokens": [SCENE_END_TOKEN, SCENE_START_TOKEN]})
        encoder.resize_token_embeddings(len(tokenizer))
        print(f"{encoder.config.vocab_size}")

        encoder = encoder.to(device)  # type: ignore
        encoder.eval()

        return tokenizer, encoder

    @staticmethod
    def load_cogvideox_model(
        job_config: JobConfig, effective_rank: int, effective_world_size: int, device: str
    ) -> CogVideoX:
        """
        Load and initialize the CogVideoX model.

        Returns:
            Initialized CogVideoX model
        """
        model_config = ModelConfig.get_preset(job_config.model.size, job_config.model.video_length, job_config)

        with torch.device("meta"):
            model = CogVideoX(model_config, effective_rank=effective_rank, effective_world_size=effective_world_size)

        # cogvideo trains with freqs buffers cast to model dtype
        cast_rotary_freqs(model, TORCH_DTYPE_MAP[job_config.parallelism.fsdp_unsharded_dtype])

        apply_parallelisms(model, job_config)

        model.to_empty(device=device)
        model.init_ssm_weights()

        state_dict = {MODEL_STATE_DICT_KEY: get_model_state_dict(model)}
        dcp.load(state_dict=state_dict, checkpoint_id=job_config.checkpoint.init_state_dir)  # type: ignore
        set_model_state_dict(model, model_state_dict=state_dict["model"], options=StateDictOptions(strict=True))

        model.eval()
        return model

    @staticmethod
    def load_vae_model(job_config: JobConfig, device: str) -> VideoAutoencoderInferenceWrapper:
        """
        Load and initialize the VAE model.

        Returns:
            Initialized VAE model
        """
        vae_checkpoint_path = job_config.eval.vae_checkpoint_path
        vae_model = VideoAutoencoderInferenceWrapper(
            vae_checkpoint_path,
            VaeModelConfig.get_encoder_config(),
            VaeModelConfig.get_decoder_config(),
            scale_factor=job_config.eval.vae_scale_factor,
        ).to(device)

        vae_model.to(TORCH_DTYPE_MAP[job_config.eval.dtype])
        vae_model.eval()
        return vae_model


class TextEncoder:
    """Handle encoding of text prompts for the model."""

    @staticmethod
    @torch.no_grad()
    def encode_text(
        tokenizer: T5Tokenizer, encoder: T5EncoderModel, prompts: List[str], device: str, maxlen: int, dtype
    ) -> torch.Tensor:
        """
        Encode text prompts using the T5 encoder.

        Returns:
            Tensor of encoded text embeddings
        """
        inputs = tokenizer(
            prompts,
            truncation=True,
            return_length=True,
            max_length=maxlen,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        with torch.autocast("cuda", enabled=False):
            outputs = encoder(input_ids=inputs["input_ids"])

        embeddings = outputs.last_hidden_state

        # Handle multi-scene vs single prompt differently
        if len(prompts) > 1:
            embeddings = embeddings.unsqueeze(0)

        return embeddings.to(dtype=dtype)


class DenoiserSampler:
    def __init__(
        self,
        model: nn.Module,
        config: JobConfig,
        dtype,
        effective_rank: int,
        seed: int = 0,
        device: str = "cuda",
        use_wandb: bool = False,
    ):
        self.sampler = VPSDEDPMPP2MSampler(
            denoiser=DiscreteDenoiser(
                model,
                num_idx=config.denoiser.num_idx,
                quantize_c_noise=config.denoiser.quantize_c_noise,
                dtype=dtype,
            ),
            discretization_config={"shift_scale": config.discretization.shift_scale},
            guider_config={
                "scale": config.guider.scale,
                "exp": config.guider.exp,
                "num_steps": config.eval.num_denoising_steps,
            },
            verbose=True,
            device=device,
            num_steps=config.eval.num_denoising_steps,
            use_wandb=use_wandb,
        )

        self.noise_generator = torch.Generator(device="cuda")
        self.noise_generator.manual_seed(effective_rank + seed)

        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def sample(self, text_emb: torch.Tensor, neg_emb: torch.Tensor, shape: tuple, batch_size: int):
        randn = torch.randn(batch_size, *shape, device=self.device, generator=self.noise_generator, dtype=torch.float32)

        cond = {
            "crossattn": text_emb
        }  # compatability with sat.sgm.modules.diffusionmodules.sampling.VPSDEDPMPP2MSampler
        uc = {
            "crossattn": neg_emb,
        }  # compatability with sat.sgm.modules.diffusionmodules.sampling.VPSDEDPMPP2MSampler

        samples = self.sampler(randn, cond, uc)
        samples = samples.to(self.dtype)
        return samples
