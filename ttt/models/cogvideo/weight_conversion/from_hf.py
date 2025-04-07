import argparse
import os
import tempfile

import torch
from safetensors import safe_open
from torch.distributed.checkpoint.format_utils import torch_save_to_dcp

from ttt.models.cogvideo.model import CogVideoX
from ttt.models.configs import ModelConfig


def main(final_save_path, ssm_layer, path_to_weights):
    # Load HuggingFace Weights
    all_tensors = {}
    file_list = [
        f"{path_to_weights}/diffusion_pytorch_model-00001-of-00002.safetensors",
        f"{path_to_weights}/diffusion_pytorch_model-00002-of-00002.safetensors",
    ]
    for filename in file_list:
        with safe_open(filename, framework="pt", device="cpu") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)

    # Create Model
    model_config = ModelConfig.get_preset("5B", "3sec")
    model_config.ssm_layer = ssm_layer
    model_config.adapter_method = "sft"
    model = CogVideoX(model_config, 0, 1).to(torch.bfloat16)

    # Convert Weights
    state_dict = {}
    for key in all_tensors:
        layer = int(key.split(".")[1]) if "transformer_blocks" in key else None

        # Patch Embeddings
        if "patch_embed.proj.bias" in key:
            state_dict["dit.patch_embedding.vid_proj.bias"] = all_tensors[key]
        elif "patch_embed.proj.weight" in key:
            state_dict["dit.patch_embedding.vid_proj.weight"] = all_tensors[key]
        elif "patch_embed.text_proj.bias" in key:
            state_dict["dit.patch_embedding.text_proj.bias"] = all_tensors[key]
        elif "patch_embed.text_proj.weight" in key:
            state_dict["dit.patch_embedding.text_proj.weight"] = all_tensors[key]

        # Transformer Norm
        elif "norm_final.bias" in key:
            state_dict["dit.transformer_norm.bias"] = all_tensors[key]
        elif "norm_final.weight" in key:
            state_dict["dit.transformer_norm.weight"] = all_tensors[key]

        # Transformer Final Layer
        elif "norm_out.norm.bias" in key:
            state_dict["dit.final_layer.norm.bias"] = all_tensors[key]
        elif "norm_out.norm.weight" in key:
            state_dict["dit.final_layer.norm.weight"] = all_tensors[key]
        elif "norm_out.linear.bias" in key:
            state_dict["dit.final_layer.adaLN_modulation.1.bias"] = all_tensors[key]
        elif "norm_out.linear.weight" in key:
            state_dict["dit.final_layer.adaLN_modulation.1.weight"] = all_tensors[key]
        elif "proj_out.bias" in key:
            state_dict["dit.final_layer.linear.bias"] = all_tensors[key]
        elif "proj_out.weight" in key:
            state_dict["dit.final_layer.linear.weight"] = all_tensors[key]

        # Time Embedding
        elif "time_embedding.linear_1.bias" in key:
            state_dict["dit.time_embed.0.bias"] = all_tensors[key]
        elif "time_embedding.linear_1.weight" in key:
            state_dict["dit.time_embed.0.weight"] = all_tensors[key]
        elif "time_embedding.linear_2.bias" in key:
            state_dict["dit.time_embed.2.bias"] = all_tensors[key]
        elif "time_embedding.linear_2.weight" in key:
            state_dict["dit.time_embed.2.weight"] = all_tensors[key]

        # Transformer Layers
        elif "transformer_blocks" in key:
            # QK Norm
            if "attn1.norm_q.bias" in key:
                state_dict[f"dit.layers.{layer}.seq_modeling_block.q_norm.bias"] = all_tensors[key]
            elif "attn1.norm_q.weight" in key:
                state_dict[f"dit.layers.{layer}.seq_modeling_block.q_norm.weight"] = all_tensors[key]
            elif "attn1.norm_k.bias" in key:
                state_dict[f"dit.layers.{layer}.seq_modeling_block.k_norm.bias"] = all_tensors[key]
            elif "attn1.norm_k.weight" in key:
                state_dict[f"dit.layers.{layer}.seq_modeling_block.k_norm.weight"] = all_tensors[key]

            # Feed Forward Network
            elif "ff.net.0.proj.bias" in key:
                state_dict[f"dit.layers.{layer}.mlp.layer1.bias"] = all_tensors[key]
            elif "ff.net.0.proj.weight" in key:
                state_dict[f"dit.layers.{layer}.mlp.layer1.weight"] = all_tensors[key]
            elif "ff.net.2.bias" in key:
                state_dict[f"dit.layers.{layer}.mlp.layer2.bias"] = all_tensors[key]
            elif "ff.net.2.weight" in key:
                state_dict[f"dit.layers.{layer}.mlp.layer2.weight"] = all_tensors[key]

            # AdaLN Norm 1 + LayerNorm 1
            elif "norm1.linear.bias" in key:
                state_dict[f"dit.layers.{layer}.pre_seq_adaLN_modulation.1.bias"] = all_tensors[key]
            elif "norm1.linear.weight" in key:
                state_dict[f"dit.layers.{layer}.pre_seq_adaLN_modulation.1.weight"] = all_tensors[key]
            elif "norm1.norm.bias" in key:
                state_dict[f"dit.layers.{layer}.pre_seq_layernorm.bias"] = all_tensors[key]
            elif "norm1.norm.weight" in key:
                state_dict[f"dit.layers.{layer}.pre_seq_layernorm.weight"] = all_tensors[key]

            # AdaLN Norm 2 + LayerNorm 2
            elif "norm2.linear.bias" in key:
                state_dict[f"dit.layers.{layer}.pre_mlp_adaLN_modulation.1.bias"] = all_tensors[key]
            elif "norm2.linear.weight" in key:
                state_dict[f"dit.layers.{layer}.pre_mlp_adaLN_modulation.1.weight"] = all_tensors[key]
            elif "norm2.norm.bias" in key:
                state_dict[f"dit.layers.{layer}.pre_mlp_layernorm.bias"] = all_tensors[key]
            elif "norm2.norm.weight" in key:
                state_dict[f"dit.layers.{layer}.pre_mlp_layernorm.weight"] = all_tensors[key]

            elif "attn1.to_q.bias" in key:
                state_dict[f"dit.layers.{layer}.seq_modeling_block.q.bias"] = all_tensors[key]
            elif "attn1.to_q.weight" in key:
                state_dict[f"dit.layers.{layer}.seq_modeling_block.q.weight"] = all_tensors[key]
            elif "attn1.to_k.bias" in key:
                state_dict[f"dit.layers.{layer}.seq_modeling_block.k.bias"] = all_tensors[key]
            elif "attn1.to_k.weight" in key:
                state_dict[f"dit.layers.{layer}.seq_modeling_block.k.weight"] = all_tensors[key]
            elif "attn1.to_v.bias" in key:
                state_dict[f"dit.layers.{layer}.seq_modeling_block.v.bias"] = all_tensors[key]
            elif "attn1.to_v.weight" in key:
                state_dict[f"dit.layers.{layer}.seq_modeling_block.v.weight"] = all_tensors[key]
            elif "attn1.to_out.0.bias" in key:
                state_dict[f"dit.layers.{layer}.seq_modeling_block.o.bias"] = all_tensors[key]
            elif "attn1.to_out.0.weight" in key:
                state_dict[f"dit.layers.{layer}.seq_modeling_block.o.weight"] = all_tensors[key]

    # Load state dict
    state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    # Save model state_dict to a temporary .pth file
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
        temp_torch_save_path = tmp_file.name

    torch.save(model.state_dict(), temp_torch_save_path)
    os.makedirs(final_save_path, exist_ok=True)

    # Convert the temporary .pth file to DCP format and save it in the DCP_DIR
    torch_save_to_dcp(temp_torch_save_path, final_save_path)
    os.remove(temp_torch_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CogVideoX HuggingFace safetensors to Torch & DCP format.")
    parser.add_argument(
        "--final_save_path", type=str, required=True, help="Directory path to save final checkpoint files."
    )
    parser.add_argument("--ssm_type", type=str, required=True, help="SSM layer type (e.g. 'ttt_mlp').")
    parser.add_argument("--pretrained_weights_dir", type=str, required=True, help="Path to the pretrained weights.")

    args = parser.parse_args()
    print(f"Starting creating new {args.ssm_type} weights at '{args.final_save_path}'...")
    main(args.final_save_path, args.ssm_type, args.pretrained_weights_dir)
    print(f"Done saving new {args.ssm_type} weights to '{args.final_save_path}'.")
