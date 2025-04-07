# Training
## Stages
Finetuning the model happens in a 5-stage curriculum based on video length.
1. 3 second (same as pretrained)
2. 9 second
3. 18 second
4. 30 second
5. 63 second

These stages are sequential, with the first stage starting from pretrained weights, and subsequent stages starting from the final checkpoint of the previous stages. The first stage (3 seconds) trains with full SFT, meaning all parameters are trainable. All other stages only train TTT parameters and the QVKO projections for local attention.

If you only need videos of a certain length, you may choose to stop training at the stage of that video length.

## Model Loading
### CogVideoX
We start our finetuning from [CogVideoX 5B](https://huggingface.co/THUDM/CogVideoX-5b) pretrained weights available on HuggingFace. We include a weight conversion script that downloads the weights and converts the model `state_dict` into the correct format.

```bash
bash scripts/convert_weights_from_hf.sh
```

You will only need to run this script for the 3 second training stage. For all later stages, you should use the final checkpoint from the end of your last training stage.

> **Note:** We do not support training from scratch, but this can be modified. Because we use pytorch [meta initialization](https://pytorch.org/docs/stable/meta.html) for FSDP and TP, all of the model weights need to be explicitly initialized once moved to the correct device. Currently, we only re-initialize some of the model weights, primarily the non-persistent buffers and rely on loading the model weights in from a checkpoint `state_dict` to initialize the rest of the parameters.

### Loading in Model State
For each stage, you will need to specify the model weights that you will begin training with. This is done through the `checkpoint.init_state_dir` config option. To specify this path, you should set the flag to the path of your checkpoint. For 3 seconds, this will be the path of the converted pretrained weights you created above. For later stages, this should be the final checkpoint from the end of your last training stage. See [`scripts/train_singlenode.sh`](../scripts/train_singlenode.sh) for an example.

## Launching Jobs

### Stages
We predefine configs corresponding to all training stages in `configs/train`. To train at a specific stage, set the `job.config_file` config option to the corresponding config file. 

Most config is preset in `configs/train` but can be overriden from the launch script. You will have to update:
- `checkpoint.init_state_dir`
  - Path to model state dict
- `training.dataset_path`
  - Path to the dataset directory
- `training.jsonl_paths`
  - Path to metadata of the dataset

> Check [here](./dataset.md) for more information on datasets.

### Local runs (Single Node)
We provide a script to run test runs on a single node. See [`scripts/train_singlenode.sh`](../scripts/train_singlenode.sh) for the template script.

### Full runs (Multi node)
We provide a script to run full training jobs on multiple nodes through slurm. We use submitit to launch our train loop on multiple nodes. See [`scripts/train_submitit.sh`](../scripts/train_submitit.sh) for the template script.

The code has implemented logic for resuming, auto-resumption, checkpointing, and logging.

> **Note**: If using auto-resume, make sure your job name does not change, as the script will look for a checkpoint for an experiment of the same name.

## Optimizations
Long context training can be challenging to fit into GPU memory due to the large size of activations. This codebase utilizes a few optimizations to allow efficient training at long context lengths. We have handled setting the required config needed for these operations in the preset configs so you should not need to worry too much about these.

### HSDP/FSDP/TP
We apply [Hybrid Sharding Data Parallel](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) (HSDP) to efficiently minimize the memory requirements of the model parameters. By using HSDP, we shard weights across devices in the same node/host, and replicate model weights across different nodes. To apply data parallel sharding (for HSDP), set the `parallelism.dp_sharding` config option to the number of devices you would like to shard weights over. To apply data parallel replicatition, set the `parallelism.dp_replicate` config option to the number of devices you would like to replicate model weights over.

For later stages, the activations of our operations become too large to maintain on a single device. We use [Tensor Parallelism](https://pytorch.org/tutorials/intermediate/TP_tutorial.html) (TP) for both acceleration and distributing memory across devices. To apply TP sharding, set the `parallelism.tp_sharding` config option to the number of devices you would like to shard over.

The product of all dimensions of your mesh (`dp_replicate * dp_sharding * tp_sharding`) should be equal to the number of gpus you are using.

> **Note:** We have set the required [device mesh](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html) needed for each stage in the preset configs, so no further work should be needed.

### Mixed Precision
We use the `mixed_precision_policy` of the pytorch FSDP wrapper to apply mixed precision. When setting the FSDP mixed precision policy to `bfloat16`, the gathered/unsharded weights and activations will be casted to `bfloat16` and all computations will be performed in this dtype. When AllReducing the gradients across devices, the reduction will be done in the higher precision of `float32`.

We rely on mixed precision for memory efficiency and speed and recommend using our default `bfloat16` dtype for training and inference.

### Rematerialization and Gradient Checkpointing
During training, activations need to be saved for the backward process of the model. However, long context activations become too large to store all at once. Our codebase uses gradient checkpointing to only store inputs to blocks of operations during the forward pass and recomputes/rematerializes the activations needed for that block only when computing the gradients for these operations. This way we minimize the total memory requirements at any given point in training, at the cost of recomputation. To learn more about gradient checkpointing, read the docs [here](https://pytorch.org/docs/stable/checkpoint.html).

> **Note:** We have config flags in the `remat` section of our config manager to apply remat for the specified portion of the code. Our preset configs already set the required remat config for that stage of training.

### Data Precomputation
To save memory and train time, we precompute the video and text embeddings prior to training. By doing this, we save the need to hold the VAE and T5 models in memory while training, maximizing the available memory for activations, and remove the need to compute the embeddings more than once. See more in [docs/dataset.md](./dataset.md).
