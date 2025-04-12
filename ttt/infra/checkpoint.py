import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

MODEL_STATE_DICT_KEY = "model"
OPTIMIZER_STATE_DICT_KEY = "optimizer"
LR_SCHEDULER_DICT_KEY = "lr_scheduler"
DATAMODULE_STATE_DICT_KEY = "data_module"
METADATA_STATE_DICT_KEY = "metadata"


class MetadataState(Stateful):
    def __init__(self, metadata=None):
        self.metadata = metadata or {}

    def state_dict(self):
        return self.metadata

    def load_state_dict(self, state_dict):
        self.metadata = state_dict


class Checkpointer:

    def __init__(self, model, optimizer, lr_scheduler, data_module, logger):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.data_module = data_module

        self.metadata = MetadataState({"wandb_id": None})

        self.logger = logger

    def _save(self, path, state_dict):
        dcp.save(state_dict=state_dict, checkpoint_id=path)

    def _load(self, path, state_dict):
        dcp.load(state_dict=state_dict, checkpoint_id=path)

    def load_pretrained(self, path):
        self.logger.write(f"Loading in state from {path}.")

        # May have more than just model state dict if from previous stage training
        try:
            state_dict = get_model_state_dict(self.model)
            self._load(path, state_dict)
        except RuntimeError:
            state_dict = {MODEL_STATE_DICT_KEY: get_model_state_dict(self.model)}
            self._load(path, state_dict)
            state_dict = state_dict[MODEL_STATE_DICT_KEY]

        set_model_state_dict(self.model, model_state_dict=state_dict, options=StateDictOptions(strict=True))

    def load(self, path):
        self.logger.write(f"Loading in state from {path}.")

        model_state_dict, optim_state_dict = get_state_dict(self.model, self.optimizer)

        state_dict = {
            MODEL_STATE_DICT_KEY: model_state_dict,
            OPTIMIZER_STATE_DICT_KEY: optim_state_dict,
            DATAMODULE_STATE_DICT_KEY: self.data_module.sampler.state_dict(),
            METADATA_STATE_DICT_KEY: self.metadata.state_dict(),
        }

        if self.lr_scheduler is not None:
            state_dict[LR_SCHEDULER_DICT_KEY] = self.lr_scheduler.state_dict()

        self._load(path, state_dict)

        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict[MODEL_STATE_DICT_KEY],
            optim_state_dict=state_dict[OPTIMIZER_STATE_DICT_KEY],
        )

        self.data_module.sampler.load_state_dict(state_dict[DATAMODULE_STATE_DICT_KEY])
        self.metadata.load_state_dict(state_dict[METADATA_STATE_DICT_KEY])

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state_dict[LR_SCHEDULER_DICT_KEY])

        self.logger.write("Completed loading in state.")

    def save(self, path):
        self.logger.write(f"Saving state at {path}.")

        model_state_dict, optim_state_dict = get_state_dict(self.model, self.optimizer)
        self.set_wandb(self.logger.wandb_logger.job_id)
        state_dict = {
            MODEL_STATE_DICT_KEY: model_state_dict,
            OPTIMIZER_STATE_DICT_KEY: optim_state_dict,
            LR_SCHEDULER_DICT_KEY: self.lr_scheduler.state_dict() if self.lr_scheduler else {},
            DATAMODULE_STATE_DICT_KEY: self.data_module.sampler.state_dict(),
            METADATA_STATE_DICT_KEY: self.metadata.state_dict(),
        }

        self._save(path, state_dict)

        self.logger.write("Completed saving state.")

    def set_wandb(self, wandb_id: str):
        self.metadata.load_state_dict({"wandb_id": wandb_id})
