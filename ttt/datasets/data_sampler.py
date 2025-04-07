from typing import Iterator

import torch
from torch.utils.data import DistributedSampler


class RandomFaultTolerantSampler(DistributedSampler):

    def __init__(self, dataset, effective_rank, effective_world_size, *args, generator=None, **kwargs):
        if generator is None:
            # We set rank dependent seeds but rely on loading the same data for ddp, just fix here
            generator = torch.Generator().manual_seed(0)

        self.generator = generator
        self.dataset = dataset
        super().__init__(dataset, *args, **kwargs)

        self.counter = 0
        self.restarting = False
        self.state = self.generator.get_state()  # Record the initial state of generator determined by seed

        self.effective_rank = effective_rank
        self.effective_world_size = effective_world_size

    def state_dict(self):
        return {"random_state": self.state, "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.generator.set_state(state_dict.get("random_state"))
        self.counter = state_dict.get("counter", 0)
        self.restarting = True

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)

        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter :]
            self.restarting = False

        for i in range(0, len(indices), self.effective_world_size):
            if i + self.effective_rank >= len(indices):
                break
            yield indices[i + self.effective_rank]
