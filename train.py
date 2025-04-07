import torch
from einops import rearrange
from torch.distributed.elastic.multiprocessing.errors import record

from ttt.datasets.preembedding_dataset import PreembeddingDataModule
from ttt.infra.checkpoint import Checkpointer
from ttt.infra.config_manager import JobConfig
from ttt.infra.logging import MultiLogger, get_logger
from ttt.infra.optimizers import get_optimizer_and_scheduler
from ttt.infra.parallelisms import (
    TORCH_DTYPE_MAP,
    apply_parallelisms,
    end_distributed,
    get_world_info,
    init_distributed,
    init_model_parameters,
)
from ttt.infra.train_iterator import TrainingIterator
from ttt.infra.utils import GarbageCollection, TimedContext, display_logo, get_num_params, get_time, set_random_seed
from ttt.models.cogvideo.model import CogVideoX
from ttt.models.cogvideo.utils import cast_rotary_freqs, dropout_txt, to_local
from ttt.models.configs import ModelConfig


def get_batch(data_iterator, dataloader, batch_size, logger):
    with TimedContext() as dataloader_time:
        try:
            batch = next(data_iterator)
        except StopIteration:
            logger.write("Resetting dataloader.")
            data_iterator = iter(dataloader)
            dataloader.sampler.counter = 0
            batch = next(data_iterator)

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()
            elif key == "txt_scene_embs":
                flattened = rearrange(batch[key], "b n s e -> b (n s) e")
                flattened = dropout_txt(flattened, 0.1)
                batch[key] = rearrange(flattened, "b (n s) e -> b n s e", n=batch[key].shape[1])

    dataloader.sampler.counter += batch_size  # Keep track for checkpointing

    return batch, data_iterator, dataloader_time.duration


@record
def main(job_config: JobConfig, logger: MultiLogger):
    get_time()  # Initialize start time for timeout checking
    gc_handler = GarbageCollection(gc_freq=job_config.training.gc_freq)

    num_steps = job_config.training.steps

    effective_rank, effective_world_size = get_world_info(job_config)

    # Each process gets a different seed for noise diversity
    set_random_seed(job_config.job.seed + effective_rank)

    # Get model config
    model_config = ModelConfig.get_preset(job_config.model.size, job_config.model.video_length, job_config)

    # Initialize skeleton of model for FSDP
    # Will allocate and init weights later
    with torch.device("meta"):
        model = CogVideoX(config=model_config, effective_rank=effective_rank, effective_world_size=effective_world_size)

    model.train()

    # CogvideoX trains with freqs buffers cast to model dtype
    if job_config.model.name == "cogvideo":
        cast_rotary_freqs(model, TORCH_DTYPE_MAP[job_config.parallelism.fsdp_unsharded_dtype])

    # Apply HSDP/FSDP/TP
    apply_parallelisms(model, job_config)

    # Allocate sharded params on gpu device and initialize
    model.to_empty(device="cuda")

    # Specifically initialize weights before loading in checkpointed weights.
    # Checkpoint loading will overwrite the necessary parameters but will have
    # the non-persistent buffers already initialized
    model.init_ssm_weights()

    data_module = PreembeddingDataModule(
        dataset_path=job_config.training.dataset_path,
        scale_factor=model_config.scale_factor,
        jsonl_paths=job_config.training.jsonl_paths,
        effective_rank=effective_rank,
        effective_world_size=effective_world_size,
    )

    optimizer, lr_scheduler, lr_scheduler_configs = get_optimizer_and_scheduler(model, job_config)

    checkpointer = Checkpointer(model, optimizer, lr_scheduler, data_module, logger)

    train_iter = TrainingIterator(
        0,  # Start step
        num_steps,
        logger,
        checkpointer,
        checkpoint_interval=job_config.checkpoint.interval,
        timeout_minutes=job_config.checkpoint.timeout_minutes,
        desc=job_config.job.exp_name,
    )

    # Choose training starting point
    is_resuming = False
    if job_config.checkpoint.resume:
        step = train_iter.resume(job_config.checkpoint.resume_step)
        if step > 0:
            is_resuming = True
            device_seed = job_config.job.seed + effective_rank + step
            model.setup_generator(seed=device_seed)
            set_random_seed(device_seed)

    if not is_resuming:
        if job_config.checkpoint.init_state_dir:  # From pretrained
            checkpointer.load_pretrained(job_config.checkpoint.init_state_dir)
        else:  # Randomly initialized
            model.apply(init_model_parameters)

    logger.init_log(job_config, model_config, get_num_params(model))

    # Must get data iterator after resuming from checkpoint
    local_bs = job_config.training.global_batch_size // effective_world_size
    dataloader = data_module.create_dataloader(local_bs, num_workers=2)
    data_iterator = iter(dataloader)

    # Start training iterations
    for step in train_iter:
        optimizer.zero_grad()
        gc_handler.run(step)  # Will garbage collect at the configured step interval

        (batch, data_iterator, dataloader_time) = get_batch(
            data_iterator, dataloader, job_config.training.global_batch_size, logger
        )

        loss = torch.tensor(0.0, device="cuda")

        # Gradient accumulation
        for accum_step in range(job_config.training.grad_accum_steps):
            micro_bs = batch["vae_emb"].shape[0] // job_config.training.grad_accum_steps

            def get_micro_batch(full_batch):
                return full_batch[accum_step * micro_bs : (accum_step + 1) * micro_bs]

            vae_emb = get_micro_batch(batch["vae_emb"])
            text_emb = get_micro_batch(batch["txt_scene_embs"])

            loss_micro = model(vae_emb, text_emb).mean() / job_config.training.grad_accum_steps

            loss_micro.backward()

            loss += loss_micro

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            job_config.optimizer.gradient_clipping_norm,
            foreach=True,
        )

        # Update weights
        optimizer.step()
        lr_scheduler.step()

        # Reduce metrics
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        grad_norm = to_local(grad_norm)  # Get local grad norm partition

        if model.tp_mesh is not None:
            grad_norm_sqrt = grad_norm.pow(2)
            torch.distributed.all_reduce(
                grad_norm_sqrt, op=torch.distributed.ReduceOp.SUM, group=model.tp_mesh.get_group(mesh_dim="tp")
            )
            grad_norm = grad_norm_sqrt.sqrt()

        train_iter.add_metric("loss", loss.item())
        train_iter.add_metric("lr", lr_scheduler.get_last_lr()[0])
        train_iter.add_metric("dataloader time", dataloader_time)
        train_iter.add_metric("grad_norm", grad_norm.item())

        lr_dict = (
            {
                f"learning_rate/{lr_scheduler_configs[i].group_name}": lr
                for i, lr in enumerate(lr_scheduler.get_last_lr())
            }
            if model_config.ssm_layer != "none"
            else {"learning_rate/remaining_wd": lr_scheduler.get_last_lr()[0]}
        )

        multi_stats = {
            "global_step": step,
            "train/loss": loss.item(),
            "gradient_norm": grad_norm.item(),
            "dataloader_time": dataloader_time,
            **lr_dict,
        }
        train_iter.add_multi(**multi_stats)


if __name__ == "__main__":
    display_logo()

    # Setup
    config = JobConfig()
    config.parse_args()
    init_distributed(config)
    logger = get_logger(config)

    # Run training
    main(config, logger)

    # Clean up processes
    end_distributed()
