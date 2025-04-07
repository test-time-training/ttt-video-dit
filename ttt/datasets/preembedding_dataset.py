import json
import os.path as osp

import decord
import torch
from torch.utils.data import Dataset

from ttt.datasets.data_sampler import RandomFaultTolerantSampler
from ttt.models.vae.regularizers import DiagonalGaussianDistribution

SCENE_END_TOKEN = "<end_scene>"
SCENE_START_TOKEN = "<start_scene>"


class PreembeddingDataset(Dataset):
    def __init__(self, dataset_path, scale_factor, jsonl_paths):
        super().__init__()

        self.dataset_path = dataset_path
        self.scale_factor = scale_factor
        self.metadata_list = []

        if isinstance(jsonl_paths, str):
            jsonl_paths = [jsonl_paths]

        for jsonl_path in jsonl_paths:
            with open(jsonl_path, "r") as f:
                for line in f:
                    metadata = json.loads(line)
                    self.metadata_list.append(metadata)

        print(f"Loaded {len(self.metadata_list)} videos")
        decord.bridge.set_bridge("torch")

    def __getitem__(self, index):
        for i in range(10):
            try:
                return self.get_data_by_index(index)
            except (TimeoutError, RuntimeError, Exception) as e:
                print(f"Error loading video {index}, retrying")
                print(e)

    def abs_path(self, path: str):
        return path if osp.isabs(path) else osp.join(self.dataset_path, path)

    def get_data_by_index(self, index):
        metadata = self.metadata_list[index]

        video_emb_path = self.abs_path(metadata["vid_emb"])
        video_emb = torch.load(video_emb_path, map_location="cpu")

        # Sample latent
        posterior = DiagonalGaussianDistribution(video_emb)
        vae_emb = posterior.sample()
        vae_emb = self.scale_factor * vae_emb

        txt_scene_embs = []
        for path in metadata["text_chunk_emb"]:
            txt_scene_emb_path = self.abs_path(path)
            txt_scene_embs.append(torch.load(txt_scene_emb_path, map_location="cpu"))

        txt_scene_embs = torch.stack(txt_scene_embs, dim=0)

        return {"vae_emb": vae_emb, "txt_scene_embs": txt_scene_embs}

    def __len__(self):
        return len(self.metadata_list)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(jsonl_paths=path, **kwargs)


class PreembeddingDataModule:
    def __init__(self, dataset_path, scale_factor, jsonl_paths, effective_rank, effective_world_size):
        self.dataset = PreembeddingDataset(dataset_path, scale_factor, jsonl_paths)
        # Important to initialize this early before creating dataloader as
        # we potential need to update the sampler when resuming from checkpoint.
        # Else, the dataloader will fork early and not receive the sampler update.
        self.sampler = RandomFaultTolerantSampler(self.dataset, effective_rank, effective_world_size)

    def create_dataloader(self, batch_size, num_workers):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
