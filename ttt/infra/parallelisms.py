import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, SequenceParallel, parallelize_module

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


TRACE_BUFFER_SIZE = "TORCH_NCCL_TRACE_BUFFER_SIZE"
TRACE_FILE = "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"
DUMP_ON_TIMEOUT = "TORCH_NCCL_DUMP_ON_TIMEOUT"
ASYNC_ERROR_HANDLING = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
SKIP_CLEANUP = "3"

type TPPlan = dict[str, ParallelStyle]


def init_distributed(job_config):
    device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
    torch.cuda.set_device(device)

    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=job_config.comm.init_timeout_seconds))

    # to mitigate the memory issue that collectives using
    # async_op=True hold memory longer than they should
    # such as those in tensor parallelism
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"


def end_distributed():
    # Sync up before exiting
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    torch.cuda.synchronize()
    torch.distributed.destroy_process_group()


def get_world_info(job_config):
    tp_sharding = job_config.parallelism.tp_sharding

    assert dist.get_world_size() % tp_sharding == 0, "world size must be divisible by tp_sharding"

    effective_rank = dist.get_rank() // tp_sharding
    effective_world_size = dist.get_world_size() // tp_sharding

    return effective_rank, effective_world_size


def get_world_mesh(job_config):
    world_size = dist.get_world_size()

    tp_sharding = job_config.parallelism.tp_sharding
    dp_sharding = job_config.parallelism.dp_sharding
    dp_replicate = job_config.parallelism.dp_replicate
    global_bs = job_config.training.global_batch_size

    assert (
        tp_sharding * dp_sharding * dp_replicate == world_size
    ), "world size must be equal to the product of tp_sharding, dp_sharding, and dp_replicate"

    assert dp_sharding <= global_bs, "dp_sharding must be greater than or equal to the global batch size"

    assert dp_replicate > 0, "dp_replicate must be greater than 0. Set to 1 to avoid data parallel on the transformer"
    assert (
        dp_sharding > 0
    ), "dp_sharding must be greater than 0. Set to 1 to avoid fully/hybrid sharding on the transformer"
    assert tp_sharding > 0, "tp_sharding must be greater than 0. Set to 1 to avoid tensor parallel on the transformer"

    if dp_sharding * dp_replicate * tp_sharding == 1:
        mesh = [dp_sharding]
        names = ["dp_shard"]

    names, mesh = [], []
    for dim, name in zip([dp_replicate, dp_sharding, tp_sharding], ["dp_replicate", "dp_shard", "tp"]):
        if dim > 1 or (name == "dp_shard" and dp_sharding == 1):
            names.append(name)
            mesh.append(dim)

    mesh = init_device_mesh("cuda", mesh, mesh_dim_names=names)

    return mesh


def apply_parallelisms(model, job_config):
    world_mesh = get_world_mesh(job_config)

    if job_config.parallelism.tp_sharding > 1:
        assert job_config.training.adapter_method is None or job_config.training.adapter_method in (
            "qkvo",
            "none",
        ), "TP is only supported with qkvo adapter method"

        apply_tp(model, world_mesh)

    apply_fsdp(model, world_mesh, job_config)


def apply_tp(model, world_mesh):
    """
    Apply tensor parallelism to the model.
    """
    tp_mesh = world_mesh["tp"]

    layer_tp_plan: TPPlan = {
        "seq_modeling_block.q": ColwiseParallel(output_layouts=Shard(2), use_local_output=False),
        "seq_modeling_block.k": ColwiseParallel(output_layouts=Shard(2), use_local_output=False),
        "seq_modeling_block.v": ColwiseParallel(output_layouts=Shard(2), use_local_output=False),
        "seq_modeling_block.o": ColwiseParallel(  # Avoid rowwise parallel here as the all-reduce has caused training divergences on torch==2.6.0
            input_layouts=Shard(-1), output_layouts=Replicate(), use_local_output=True
        ),
        "seq_modeling_block.q_norm": SequenceParallel(sequence_dim=1, use_local_output=False),
        "seq_modeling_block.k_norm": SequenceParallel(sequence_dim=1, use_local_output=False),
        "seq_modeling_block.ssm.ttt.wq": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
        "seq_modeling_block.ssm.ttt.wk": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
        "seq_modeling_block.ssm.ttt.wv": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
        "seq_modeling_block.ssm.ttt.post_norm": SequenceParallel(sequence_dim=1),
        "seq_modeling_block.ssm.ttt.wo": ColwiseParallel(
            use_local_output=False
        ),  # Avoid rowwise parallel here as the all-reduce has caused training divergences on torch==2.6.0
    }

    model_tp_plan: TPPlan = {
        "final_layer.linear": ColwiseParallel(output_layouts=Replicate(), use_local_output=True),
    }

    for transformer_block in model.dit.layers:
        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )

    parallelize_module(
        module=model.dit,
        device_mesh=tp_mesh,
        parallelize_plan=model_tp_plan,
    )

    # parallelize_module enables requires_grad
    model.dit.final_layer.linear.requires_grad_(False)

    model.init_device_mesh(tp_mesh)

    return tp_mesh


def apply_fsdp(model, world_mesh, job_config):
    param_dtype = TORCH_DTYPE_MAP[job_config.parallelism.fsdp_unsharded_dtype]
    reduce_dtype = torch.float32  # Anything less than fp32 significantly hurts numerically

    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    dp_mesh = (
        world_mesh[("dp_replicate", "dp_shard")] if job_config.parallelism.dp_replicate > 1 else world_mesh["dp_shard"]
    )

    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    # Don't shard outside of the DiT.
    # Would add communication costs that probably outweigh the computation
    # being done.
    for _, layer in model.dit.layers.named_children():
        fully_shard(
            layer,
            **fsdp_config,
            reshard_after_forward=True,
        )
    fully_shard(model.dit, **fsdp_config, reshard_after_forward=True)


def init_model_parameters(model, initializer_range=0.02):
    from ttt.models.ssm.ttt_layer import TTTBase

    for module in model.modules():
        if isinstance(module, TTTBase):
            # Let the TTTBase module handle its own initialization
            # TTT has a complicated initialization
            module.init_weights()
            continue

        for name, param in module.named_parameters(recurse=False):
            # Check that the parameter is actually defined (not None)
            if param is None:
                continue

            if "bias" in name:
                torch.nn.init.zeros_(param)
            else:
                torch.nn.init.normal_(param, mean=0.0, std=initializer_range)
