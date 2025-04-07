import argparse
import copy
import os
import subprocess
from datetime import datetime

import submitit

import train
from ttt.infra.config_manager import JobConfig
from ttt.infra.logging import get_logger
from ttt.infra.parallelisms import end_distributed, init_distributed


def parse_args(input_args=None):
    parser = argparse.ArgumentParser("Submitit for CogVideo training")
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=8, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=20160, type=int, help="Duration of the job")
    parser.add_argument("--slurm_max_num_timeout", default=15, type=int, help="maximum number of retries")
    parser.add_argument(
        "--account",
        "-A",
        default="default",
        type=str,
        help="Slurm account",
    )
    parser.add_argument(
        "--partition",
        default="default",
        type=str,
        help="Partition where to submit",
    )
    parser.add_argument("--nodelist", default=None, type=str, help="specify node list")
    parser.add_argument(
        "--comment",
        default="",
        type=str,
        help="Comment to pass to scheduler, e.g. priority message",
    )
    parser.add_argument("--dependency_job_id", default="", type=str)
    args, remaining = parser.parse_known_args(input_args)
    return args, remaining


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        self._setup_gpu_args()
        args = copy.deepcopy(self.args)

        self._log_env()

        init_distributed(args.job_config)
        logger = get_logger(args.job_config)
        train.main(args.job_config, logger)
        end_distributed()

    def _log_env(self):
        try:
            git_commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
            print(f"Git commit hash: {git_commit_hash}")
        except Exception as e:
            print(f"Could not retrieve git commit hash: {e}")

    def checkpoint(self, *args, **kwargs):
        """Save any state that we want available when requeuing happens"""

        delayed_sub = submitit.helpers.DelayedSubmission(self)
        delayed_sub.set_timeout(timeout_min=5, max_num_timeout=3)

        return delayed_sub

    def _setup_gpu_args(self):
        # Handles the environment variable setting that torchrun typically sets
        dist_env = submitit.helpers.TorchDistributedEnvironment().export(set_cuda_visible_devices=False)
        print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        print(f"rank: {dist_env.rank}")
        print(f"world size: {dist_env.world_size}")
        print(f"local rank: {dist_env.local_rank}")
        print(f"local world size: {dist_env.local_world_size}")


def main(input_args=None):
    args, remaining = parse_args(input_args)
    job_config = JobConfig()
    job_config.parse_args(remaining)
    args.job_config = job_config

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    slurm_max_num_timeout = args.slurm_max_num_timeout
    kwargs = {}
    if args.comment:
        kwargs["slurm_comment"] = args.comment

    output_dir = job_config.job.dump_folder
    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(
        folder=os.path.join(
            output_dir,
            "submitit_logs",
            f"%j-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        ),
        slurm_max_num_timeout=slurm_max_num_timeout,
    )

    executor.update_parameters(
        mem_gb=1400,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=15,
        nodes=nodes,
        timeout_min=timeout_min,
        # Below are cluster dependent parameters
        slurm_account=args.account,
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        stderr_to_stdout=True,
        **kwargs,
    )

    trainer = Trainer(args)
    job = executor.submit(trainer)
    print("Submitted job_id: ", job.job_id)


if __name__ == "__main__":
    main()
