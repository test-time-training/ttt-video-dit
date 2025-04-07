import datetime
import gc
import random
import time

import numpy as np
import torch


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True  # Can set to False if needed
    torch.backends.cuda.matmul.allow_tf32 = False  # Faster but less accurate if set to True


class TimedContext:
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time


def get_time():
    if not hasattr(get_time, "cached_time"):
        get_time.cached_time = datetime.datetime.now()  # type: ignore
    return get_time.cached_time  # type: ignore


def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= model.tok_embeddings.weight.numel()  # type: ignore
    return num_params


# used to avoid stragglers in garbage collection
class GarbageCollection:
    def __init__(self, gc_freq: int = 1000):
        assert gc_freq > 0, "gc_freq must be a positive integer"
        self.gc_freq = gc_freq
        gc.disable()
        gc.collect(1)

    def run(self, step_count: int):
        if step_count > 1 and step_count % self.gc_freq == 0:
            gc.collect(1)


def display_logo():
    # fmt: off
    print(r"""
   /$$$$$$$$ /$$$$$$  /$$      /$$        /$$$              /$$$$$ /$$$$$$$$ /$$$$$$$  /$$$$$$$  /$$     /$$
  |__  $$__//$$__  $$| $$$    /$$$       /$$ $$            |__  $$| $$_____/| $$__  $$| $$__  $$|  $$   /$$/
     | $$  | $$  \ $$| $$$$  /$$$$      |  $$$                | $$| $$      | $$  \ $$| $$  \ $$ \  $$ /$$/ 
     | $$  | $$  | $$| $$ $$/$$ $$       /$$ $$/$$            | $$| $$$$$   | $$$$$$$/| $$$$$$$/  \  $$$$/  
     | $$  | $$  | $$| $$  $$$| $$      | $$  $$_/       /$$  | $$| $$__/   | $$__  $$| $$__  $$   \  $$/   
     | $$  | $$  | $$| $$\  $ | $$      | $$\  $$       | $$  | $$| $$      | $$  \ $$| $$  \ $$    | $$    
     | $$  |  $$$$$$/| $$ \/  | $$      |  $$$$/$$      |  $$$$$$/| $$$$$$$$| $$  | $$| $$  | $$    | $$    
     |__/   \______/ |__/     |__/       \____/\_/       \______/ |________/|__/  |__/|__/  |__/    |__/    

                                                                    Coming soon to a theater near you...                                                   
""")
    # fmt: on
