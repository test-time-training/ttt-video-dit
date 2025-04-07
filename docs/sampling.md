# Sampling

This guide explains how to generate videos from text prompts using the pretrained model with the provided sampling script, `sample.py`.

## Overview

Sampling is the process of generating videos from text prompts using the trained diffusion transformer model. Our sampling script (`sample.py`) utilizes Test-Time Training (TTT) layers to produce coherent, multi-scene videos up to one minute in length directly from storyboards.

During sampling, the model takes text input structured into scenes and segments and progressively generates video frames by iteratively denoising random noise tensors guided by the input prompts. The diffusion process leverages classifier-free guidance and can incorporate negative conditioning to enhance video coherence and scene transitions.

## Sampling Pipeline

At a high level, the sampling pipeline consists of:  

- Parsing textual prompts into structured segments.  
- Encoding prompts into text embeddings using T5.  
- Iteratively denoising latent representations guided by prompts and negative conditions.  
- Decoding latents into video frames using a VAE decoder.  

## Sampling Mechanics

### Diffusion Process

The sampling process works by reversing the diffusion process used during training:

1. **Initialization**: The process begins with a randomly sampled noise tensor (representing the initial video frames).
2. **Denoising Steps**: The model progressively denoises this tensor through multiple steps.
3. **Guidance**: Text prompts guide the generation to align with the desired content.

### Scheduler

The sampling process uses a noise scheduler to control the denoising trajectory:

- Our implementation uses a v-prediction [DDIM sampler](https://arxiv.org/abs/2010.02502) with ZeroSNR enforced at the final step.
  - The scheduler determines how quickly noise is removed at each step.
  - For standard sampling, we use 50 denoising steps.

### Classifier-Free Guidance

Guidance helps steer the generation toward the text prompt:

- Dynamic classifier-free guidance is applied, with guidance scale increasing from 1 to 6 during sampling.
- The guidance scale controls how strictly the generation follows the text prompt.
- Higher guidance scales result in stronger text alignment but may produce less diverse or natural videos.

### Negative Conditioning

Negative prompts help improve generation quality by specifying undesired elements:

- Each scene can have an optional negative prompt (`neg_text`) to avoid specific attributes.
- The model processes both the positive and negative conditions, using the guidance scale to steer away from negative attributes.
- This approach helps avoid common generation issues and improves overall quality.

## Basic Usage

```bash
torchrun --nproc_per_node=<NUM_GPUS> \
  --rdzv_backend c10d \
  --rdzv_endpoint="localhost:0" \
  --local-ranks-filter 0 \
  --role rank \
  --tee 3 \
  sample.py \
  --job.config_file <CONFIG_FILE> \
  --checkpoint.init_state_dir=<CHECKPOINT_DIR> \
  [additional options]
```

## Configuration

### Required Parameters

- `--job.config_file`: Path to the TOML configuration file.
- `--checkpoint.init_state_dir`: Directory containing model weights.
- `--eval.t5_model_dir`: Directory containing the T5 text encoder model.
- `--eval.vae_checkpoint_path`: Path to the VAE checkpoint.

### Input Methods

The script accepts JSON or JSONL files via the `--eval.input_file` parameter:

- **JSON/JSONL Format**:
 - Use `--eval.input_file=<path>` to specify a JSON or JSONL file.
 - For JSONL files, each line contains a separate video description.
 - Each video description is an array of scene objects with the structure:
   ```json
   [
     {
       "text": "First scene description",
       "requires_scene_transition": false,
       "neg_text": "Optional negative prompt for this scene"
     },
     {
       "text": "Second scene description",
       "requires_scene_transition": true,
       "neg_text": "Optional negative prompt for this scene"
     }
   ]
   ```
 
 - **Field Descriptions**:
   - `text`: The scene description/prompt (required).
   - `requires_scene_transition`: Boolean flag indicating if this scene needs a transition from the *previous* scene (default: false).
   - `neg_text`: Optional negative prompt for this specific scene.

See a complete example [here](../inputs/example-9s.json).

### Output Options

- `--eval.output_dir`: Output directory for generated videos (default: `./output`)
- `--eval.image_width`: Width of generated video (default: 720)
- `--eval.image_height`: Height of generated video (default: 480)
- `--eval.sampling_fps`: Frames per second in output video (default: 16)
- `--eval.sampling_num_frames`: Number of frames to generate (default: 13)

### Sampling Parameters

- `--eval.num_denoising_steps`: Number of denoising steps (default: 50)
- `--guider.scale`: Guidance scale (default: 6)
- `--eval.dtype`: Datatype for sampling (options: bfloat16, float16, float32)

### Video Length Options

Based on the research paper, the model supports generating videos of different lengths:
1. 3 second (same as pretrained)
2. 9 second
3. 18 second
4. 30 second
5. 63 second

To generate videos of specific lengths, ensure you're using the checkpoint trained on that stage or longer.

### Parallelism Options
> To enable parallelism, set the following parameters to a value greater than 1.

- `--parallelism.dp_sharding=<int>`: Use data parallel sharding.
- `--parallelism.dp_replicate=<int>`: Use data parallel replication.
- `--parallelism.tp_sharding=<int>`: Use tensor parallel sharding.

### Multi-Scene Generation

For longer videos with multiple scenes, the model uses a specialized approach:
- Each scene is processed in 3-second segments.
- The model applies local attention within each segment.
- Test-Time Training (TTT) layers handle global context across segments.
- Scene transitions are marked with special tokens (`<scene_start>` and `<scene_end>`). These are automatically inserted based on the `requires_scene_transition` flag in the input JSON(L) file.

## Example Scripts

See `./scripts/sample_singlenode.sh` for a sample script for generating videos on a single node.

## Tips

- To track generations with Weights & Biases, remove `--wandb.disable` and set your API key.
- Prompts should be as close as possible to the diction, syntax, and structure of the training data.
- Use negative prompts to improve generation quality by specifying unwanted elements.
