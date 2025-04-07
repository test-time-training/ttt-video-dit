import logging
import os
import os.path as osp
from datetime import datetime
from functools import partial

import pytz
import torch
import wandb

from ttt.infra.config_manager import JobConfig
from ttt.models.configs import ModelConfig

logger = logging.getLogger()

_multi_logger = None


class WandBLogger:
    def __init__(self, job_config: JobConfig):
        # Only log on one rank
        self.enable = not job_config.wandb.disable
        self.is_main = torch.distributed.get_rank() == 0

        self.job_id = None

    def initialize(self, job_config: JobConfig):
        if not self.enable:
            return

        # If is main then initialize WandB job
        if self.is_main:
            # Resume job
            if self.job_id is not None:
                wandb.init(
                    project=job_config.wandb.project,
                    entity=job_config.wandb.entity,
                    id=self.job_id,
                    resume="allow",
                    config=job_config.to_dict(),
                )
                assert wandb.run is not None, "WandB run failed to initialize."

                if job_config.wandb.alert:
                    wandb.run.alert(
                        title=f"{job_config.job.exp_name} resuming",
                        text=f"Resuming '{job_config.job.exp_name}' at wandb_id f{self.job_id}.",
                    )
            else:
                wandb.init(
                    project=job_config.wandb.project,
                    entity=job_config.wandb.entity,
                    name=job_config.job.exp_name,
                    config=job_config.to_dict(),
                    resume="allow",
                )
                assert wandb.run is not None, "WandB run failed to initialize."
                if job_config.wandb.alert:
                    wandb.run.alert(
                        title=f"{job_config.job.exp_name} launched",
                        text=f"Starting '{job_config.job.exp_name}'",
                    )
                self.job_id = wandb.run.id

        obj_list = [self.job_id]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        self.job_id = obj_list[0]

    def log(self, stats, step):
        if self.enable:
            wandb.log(stats, step=step)


class MultiLogger:
    def __init__(self, job_config: JobConfig):
        self.enable = torch.distributed.get_rank() == 0
        init_logger()

        self.exp_name = job_config.job.exp_name
        self.multi_dir = osp.join(job_config.job.dump_folder, job_config.job.exp_name)
        self.file_name = get_unique_log_file(osp.join(self.multi_dir, "log.txt"))

        self.wandb_logger = WandBLogger(job_config)

        os.makedirs(self.multi_dir, mode=0o777, exist_ok=True)

        if self.enable:
            self.local_logger = open(self.file_name, "a")

    def write(self, text: str, log_terminal: bool = True):
        if log_terminal:
            logger.info(text)

        if self.enable:
            self.local_logger.write(text + "\n")
            self.local_logger.flush()

    def init_log(self, job_config: JobConfig, model_config: ModelConfig, param_count: int):
        self.wandb_logger.initialize(job_config)

        timezone = pytz.timezone("America/Los_Angeles")
        formatted_current_time = datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S %Z%z")

        # Don't log to terminal
        file_only_write = partial(self.write, log_terminal=False)

        file_only_write(
            f"==================================\n"
            f"Launching Time: {formatted_current_time}\n"
            f"==================================\n",
        )

        self.write(f"Starting job: {job_config.job.exp_name}\n")

        file_only_write("============= Training Config ===============\n")
        file_only_write(f"{job_config}\n\n")

        self.write("============= Model Config ===============\n", log_terminal=False)
        self.write(
            f"Model Name: {job_config.model.name}\nSize: {job_config.model.size}\nVideo Length: {job_config.model.video_length}\nParam count: {param_count:,}\n{model_config}\n\n"
        )

        # Get ready for training logs
        self.write("============= Training ===============\n", log_terminal=False)

    def save_multi(self, stats, step, should_checkpoint=False):
        if not self.enable:
            return

        torch.save(stats, osp.join(self.multi_dir, "all_stat_dict.pth"))

        self.wandb_logger.log(get_last_values(stats), step=step - 1)

        if should_checkpoint:
            checkpoint_dir = get_checkpoint_dir(self.multi_dir, step)
            os.makedirs(checkpoint_dir, mode=0o777, exist_ok=True)
            torch.save(stats, osp.join(checkpoint_dir, "all_stat_dict.pth"))

    def load_multi(self, step: int | str):
        checkpoint_dir = get_checkpoint_dir(self.multi_dir, step)
        file_path = osp.join(checkpoint_dir, "all_stat_dict.pth")

        if not osp.exists(file_path):
            self.write("WARNING: Resuming without finding multi stats.\n")
            return {}

        return torch.load(file_path)

    def get_latest_checkpoint_step(self):
        """Find the latest checkpoint step in the experiment directory."""
        checkpoint_dir = os.path.join(self.multi_dir, "checkpoint")
        if not os.path.exists(checkpoint_dir):
            return None

        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("step-")]
        if not checkpoints:
            return None

        # Extract step numbers and find max
        steps = [int(cp.split("-")[1]) for cp in checkpoints]
        return max(steps)

    def close(self):
        if self.enable:
            self.local_logger.close()

        if wandb.run:
            wandb.finish()


def init_logger():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"


def get_logger(job_config: JobConfig) -> MultiLogger:
    global _multi_logger
    if _multi_logger is None:
        _multi_logger = MultiLogger(job_config)

    return _multi_logger


def get_unique_log_file(base_name: str):
    if not osp.exists(base_name):
        return base_name

    base, ext = osp.splitext(base_name)
    counter = 1
    new_name = f"{base}_{counter}{ext}"

    # Loop until a unique file name is found
    while osp.exists(new_name):
        counter += 1
        new_name = f"{base}_{counter}{ext}"

    return new_name


def get_checkpoint_dir(base_name: str, step: int | str):
    return osp.join(base_name, "checkpoint", f"step-{step}")


def get_last_values(dict_of_arrays):
    result = {}
    for key, arr in dict_of_arrays.items():
        if not arr:  # Skip or handle empty arrays as needed
            result[key] = None
            continue

        last_value = arr[-1]
        if torch.is_tensor(last_value):
            last_value = last_value.item()
        result[key] = last_value
    return result
