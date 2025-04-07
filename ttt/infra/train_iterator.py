import os
import time
from datetime import datetime, timedelta

import torch

from ttt.infra.checkpoint import Checkpointer
from ttt.infra.logging import MultiLogger, get_checkpoint_dir
from ttt.infra.utils import get_time


class TrainingIterator:
    def __init__(
        self,
        start_step,
        total_steps,
        logger: MultiLogger,
        checkpointer: Checkpointer,
        checkpoint_interval=0,
        timeout_minutes=0,
        desc=None,
    ):
        self.logger = logger
        self.checkpointer = checkpointer
        self.checkpoint_interval = checkpoint_interval
        self.desc = desc

        self.start_step = start_step
        self.steps = range(start_step, total_steps)

        self.total = total_steps
        self.current = start_step
        self.metrics = {}
        self.multi_stats = {}

        self.last_time = None
        self.avg_iter_time = None

        self.timeout_checkpoint = False
        self.timeout_minutes = timeout_minutes

    def add_metric(self, key, value):
        self.metrics[key] = value

    def add_multi(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.multi_stats:
                self.multi_stats[key] = []
            self.multi_stats[key].append(value)

    def resume(self, step):
        """Resume training from a checkpoint.

        Args:
            step: Step to resume from. If -1, will auto-detect latest checkpoint.
        """
        if step == -1:
            step = self.logger.get_latest_checkpoint_step()
            if step is None:
                self.logger.write("No checkpoints found, starting from beginning.")
                return 0

        self.logger.write(f"Resuming experiment at step {step}.")

        checkpoint_dir = get_checkpoint_dir(self.logger.multi_dir, step)
        self.checkpointer.load(checkpoint_dir)

        self.multi_stats = self.logger.load_multi(step)

        # Make sure the steps resume
        self.current = step
        self.start_step = step
        self.steps = range(step, self.total)

        self.logger.wandb_logger.job_id = self.checkpointer.metadata.state_dict()["wandb_id"]

        return step

    def __iter__(self):
        self.steps_iter = iter(self.steps)
        return self

    def __next__(self):
        # Wrap up last step
        self._time_step()
        self._print_progress()
        self.logger.save_multi(self.multi_stats, self.current)

        if self._should_checkpoint():
            self._checkpoint()

        # Mark start of next step
        try:
            item = next(self.steps_iter)
            self.current += 1
            return item
        except StopIteration:
            self._finish()
            raise

    def _time_step(self):
        current_time = time.perf_counter()
        if self.last_time is not None:
            step_duration = current_time - self.last_time

            # Throw out first time since it usually is an outlier
            if self.avg_iter_time is None or self.current == self.start_step + 2:
                self.avg_iter_time = step_duration  # Initialize EMA with first step
            else:
                # TQDM alpha is 0.3
                self.avg_iter_time = (1 - 0.3) * self.avg_iter_time + 0.3 * step_duration
        else:
            self.start_time = current_time
        self.last_time = current_time  # Update last iteration time

    def _should_checkpoint(self):
        if self.checkpoint_interval == 0 or self.current == self.steps[0]:
            return False

        # Only check if timeout_checkpoint is not already enabled
        if self.timeout_minutes != 0 and self.timeout_checkpoint is False:
            overall_start_time = get_time()
            current_time = datetime.now()
            elapsed_time = current_time - overall_start_time

            # Ignore the first step since it is usually an outlier
            if self.avg_iter_time is not None and self.current > self.start_step + 2:
                # Calculate the effective threshold: Timeout minutes - (average iteration time + 2 minutes)
                avg_iter_td = timedelta(seconds=self.avg_iter_time)
                effective_threshold = timedelta(minutes=self.timeout_minutes) - (avg_iter_td + timedelta(minutes=6))

                is_past_threshold = int(elapsed_time >= effective_threshold)
                is_past_threshold_tensor = torch.tensor(
                    [1 if is_past_threshold else 0], dtype=torch.uint8, device="cuda"
                )
                torch.distributed.all_reduce(is_past_threshold_tensor, op=torch.distributed.ReduceOp.MAX)
                is_past_threshold = bool(is_past_threshold_tensor.item())

                if is_past_threshold:
                    self.timeout_checkpoint = True  # Important
                    self.logger.write("Approaching timeout, saving checkpoint...")
                    return True

        return self.current % self.checkpoint_interval == 0 or self.current == self.total

    def _print_progress(self):
        # Construct progress message
        progress_message = f"{self.desc}: " if self.desc else ""
        progress_message += f"Steps {self.current}/{self.total}"

        if self.avg_iter_time is not None:
            # Calculate remaining steps
            remaining_steps = self.total - self.current
            time_left = timedelta(seconds=int(self.avg_iter_time * remaining_steps))

            progress_message += f" | <s/iter: {self.avg_iter_time:.2f}s"
            progress_message += f" | est: {str(time_left)}>"

        # Append metrics if any
        if self.metrics:
            metrics_str = ", ".join([f"{k}: {v:.8f}" for k, v in self.metrics.items()])
            progress_message += f" | Metrics: {metrics_str}"

        self.logger.write(progress_message)

    def _finish(self):
        total_time = time.perf_counter() - self.start_time
        total_time_str = f"{total_time:.2f}s"

        completion_message = f"{self.desc}: " if self.desc else ""
        completion_message += f"Completed {self.total} steps in {total_time_str}!"

        self.logger.write(completion_message)
        self.logger.close()
        del self.logger

    def __del__(self):
        if self.logger:
            self.logger.close()

    def _checkpoint(self):
        self.logger.write(f"Checkpointing at step {self.current}.")
        self.logger.save_multi(self.multi_stats, self.current, should_checkpoint=True)

        checkpoint_path = os.path.join(self.logger.multi_dir, "checkpoint", f"step-{self.current}")

        self.checkpointer.save(checkpoint_path)
        self.logger.write("Finished checkpointing, resuming...")
