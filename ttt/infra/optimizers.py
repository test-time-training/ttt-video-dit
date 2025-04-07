import functools
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


class ScheduleType(str, Enum):
    """Learning rate schedule types"""

    COSINE = "cosine"
    LINEAR = "linear"


@dataclass(frozen=True)
class ScheduleConfig:
    """Configuration for learning rate schedules"""

    schedule_type: ScheduleType
    warmup_steps: int
    total_steps: int
    lr_peak: float
    lr_end: float
    group_name: str


class ParameterGroupManager:
    """Manages model parameter grouping for optimization"""

    NO_WEIGHT_DECAY_PATTERNS = ["bias", "norm", "b1", "b2"]
    TTT_PARAMETER_PATTERNS = ["ttt", "ssm"]

    WEIGHT_DECAY_VALUE = 1e-4

    @classmethod
    def is_ttt_parameter(cls, param_name: str) -> bool:
        """Determine if parameter belongs to TTT/SSM group"""
        return any(pattern in param_name.lower() for pattern in cls.TTT_PARAMETER_PATTERNS)

    @classmethod
    def should_skip_weight_decay(cls, param_name: str) -> bool:
        """Determine if parameter should skip weight decay"""
        return any(pattern in param_name.lower() for pattern in cls.NO_WEIGHT_DECAY_PATTERNS)

    @staticmethod
    def create_param_group(params: List[nn.Parameter], lr: float, weight_decay: float) -> Dict[str, Any]:
        """Create a parameter group dictionary for optimizer"""
        if not params:
            raise ValueError("No parameters found for the group")

        return {"params": params, "lr": lr, "weight_decay": weight_decay}

    @classmethod
    def categorize_parameters(cls, model: nn.Module) -> Tuple[List, List, List, List]:
        """
        Categorize model parameters into four groups:
        1. TTT parameters without weight decay
        2. TTT parameters with weight decay
        3. Other parameters without weight decay
        4. Other parameters with weight decay
        """
        ttt_no_wd = []
        ttt_with_wd = []
        other_no_wd = []
        other_with_wd = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            is_ttt = cls.is_ttt_parameter(name)
            skip_wd = cls.should_skip_weight_decay(name)

            if is_ttt:
                if skip_wd:
                    ttt_no_wd.append(param)
                else:
                    ttt_with_wd.append(param)
            else:
                if skip_wd:
                    other_no_wd.append(param)
                else:
                    other_with_wd.append(param)

        return ttt_no_wd, ttt_with_wd, other_no_wd, other_with_wd


def create_optimizer(model: nn.Module, learning_rate: float) -> torch.optim.AdamW:
    """
    Create an AdamW optimizer with two parameter groups:
    1. Parameters with no weight decay
    2. Parameters with weight decay

    Args:
        model: PyTorch model
        learning_rate: Base learning rate for all parameters

    Returns:
        Configured AdamW optimizer
    """
    no_weight_decay_params = []
    weight_decay_params = []

    # Separate parameters based on whether they should have weight decay
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if ParameterGroupManager.should_skip_weight_decay(name):
            no_weight_decay_params.append(param)
        else:
            weight_decay_params.append(param)

    # Create parameter groups
    param_groups = [
        {"params": no_weight_decay_params, "weight_decay": 0.0},
        {"params": weight_decay_params, "weight_decay": ParameterGroupManager.WEIGHT_DECAY_VALUE},
    ]

    # AdamW optimizer settings
    optimizer_settings = {
        "lr": learning_rate,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
    }

    return torch.optim.AdamW(param_groups, **optimizer_settings)


def _create_param_groups_for_adapter(
    adapter_method: str,
    ttt_no_wd: List[nn.Parameter],
    ttt_with_wd: List[nn.Parameter],
    other_no_wd: List[nn.Parameter],
    other_with_wd: List[nn.Parameter],
    ssm_lr: float,
    base_lr: float,
    final_lr: float,
    ssm_lr_schedule: ScheduleType,
    base_lr_schedule: ScheduleType,
    warmup_steps: int,
    total_steps: int,
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with schedules based on adapter method

    Args:
        adapter_method: Type of adapter method (sft, qkvo)
        ttt_no_wd: TTT parameters without weight decay
        ttt_with_wd: TTT parameters with weight decay
        other_no_wd: Other parameters without weight decay
        other_with_wd: Other parameters with weight decay
        ssm_lr: Learning rate for SSM/TTT parameters
        base_lr: Base learning rate for other parameters
        final_lr: Final learning rate after decay
        ssm_lr_schedule: Schedule type for SSM parameters
        base_lr_schedule: Schedule type for base parameters
        warmup_steps: Number of warmup steps
        total_steps: Total training steps

    Returns:
        List of parameter groups with their schedules
    """

    assert adapter_method in ["sft", "qkvo"], f"Unsupported adapter method: '{adapter_method}'"
    pg_manager = ParameterGroupManager
    wd_value = pg_manager.WEIGHT_DECAY_VALUE

    ttt_groups = [
        {
            "param_group": pg_manager.create_param_group(ttt_no_wd, ssm_lr, 0.0),
            "lr_schedule": ScheduleConfig(ssm_lr_schedule, warmup_steps, total_steps, ssm_lr, final_lr, "ttt_no_wd"),
        },
        {
            "param_group": pg_manager.create_param_group(ttt_with_wd, ssm_lr, wd_value),
            "lr_schedule": ScheduleConfig(ssm_lr_schedule, warmup_steps, total_steps, ssm_lr, final_lr, "ttt_wd"),
        },
    ]

    remaining_groups = [
        {
            "param_group": pg_manager.create_param_group(other_no_wd, base_lr, 0.0),
            "lr_schedule": ScheduleConfig(
                base_lr_schedule, warmup_steps, total_steps, base_lr, final_lr, "other_no_wd"
            ),
        },
        {
            "param_group": pg_manager.create_param_group(other_with_wd, base_lr, wd_value),
            "lr_schedule": ScheduleConfig(base_lr_schedule, warmup_steps, total_steps, base_lr, final_lr, "other_wd"),
        },
    ]

    return ttt_groups + remaining_groups


def create_specialized_optimizer(
    model: nn.Module,
    base_lr: float,
    ssm_lr: float,
    final_lr: float,
    warmup_steps: int,
    total_steps: int,
    base_lr_schedule: ScheduleType,
    ssm_lr_schedule: ScheduleType,
    adapter_method: str,
) -> Tuple[torch.optim.AdamW, List[ScheduleConfig]]:
    """
    Create an optimizer with specialized parameter groups for different model components.

    Args:
        model: PyTorch model
        base_lr: Base learning rate for most parameters
        ssm_lr: Learning rate for SSM/TTT parameters
        final_lr: Final learning rate after decay
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        base_lr_schedule: Schedule type for base parameters
        ssm_lr_schedule: Schedule type for SSM parameters
        adapter_method: Adaptation method (sft, qkvo)

    Returns:
        Tuple of (optimizer, schedule configurations)
    """
    # Get categorized parameters
    ttt_no_wd, ttt_with_wd, other_no_wd, other_with_wd = ParameterGroupManager.categorize_parameters(model)

    # Check if we have any parameters to train
    if not any([ttt_no_wd, ttt_with_wd, other_no_wd, other_with_wd]):
        raise ValueError("No trainable parameters found in the model")

    # Create parameter groups based on adapter method
    param_groups_with_schedule = _create_param_groups_for_adapter(
        adapter_method,
        ttt_no_wd,
        ttt_with_wd,
        other_no_wd,
        other_with_wd,
        ssm_lr,
        base_lr,
        final_lr,
        ssm_lr_schedule,
        base_lr_schedule,
        warmup_steps,
        total_steps,
    )

    # Extract param groups and schedules
    param_groups = [pg["param_group"] for pg in param_groups_with_schedule if pg["param_group"] is not None]
    lr_schedules = [pg["lr_schedule"] for pg in param_groups_with_schedule if pg["param_group"] is not None]

    if not param_groups:
        raise ValueError("No valid parameter groups created")

    # Create optimizer with AdamW settings
    optimizer_settings = {
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }

    return torch.optim.AdamW(param_groups, **optimizer_settings), lr_schedules


class LRScheduleFunctions:
    """Learning rate schedule calculation functions"""

    @staticmethod
    def cosine_decay_with_warmup(
        warmup_steps: int,
        decay_steps: int,
        lr_peak: float,
        lr_end: float,
        current_step: int,
    ) -> float:
        """
        Calculate learning rate multiplier for cosine decay with warmup

        Args:
            warmup_steps: Number of warmup steps
            decay_steps: Number of decay steps
            lr_peak: Peak learning rate
            lr_end: Final learning rate
            current_step: Current training step

        Returns:
            Learning rate multiplier
        """
        # Handle case where both lr values are 0 (frozen parameters)
        if lr_peak == 0 and lr_end == 0:
            return 1.0

        if current_step < warmup_steps:
            # Linear warmup
            current_step += 1  # Adjust for 0-indexed step count
            return float(current_step / warmup_steps)
        else:
            # Cosine decay
            step_in_decay = current_step - warmup_steps
            cosine_factor = 0.5 * (1 + math.cos(math.pi * step_in_decay / decay_steps))
            return (lr_end + (lr_peak - lr_end) * cosine_factor) / lr_peak

    @staticmethod
    def linear_decay_with_warmup(
        warmup_steps: int, total_steps: int, lr_peak: float, lr_end: float, current_step: int
    ) -> float:
        """
        Calculate learning rate multiplier for linear decay with warmup

        Args:
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            lr_peak: Peak learning rate
            lr_end: Final learning rate
            current_step: Current training step

        Returns:
            Learning rate multiplier
        """
        if current_step < warmup_steps:
            # Linear warmup
            current_step += 1
            return float(current_step / warmup_steps)
        else:
            # Linear decay
            step_in_decay = current_step - warmup_steps
            decay_steps = max(1, total_steps - warmup_steps)
            fraction = min(step_in_decay / decay_steps, 1.0)
            return 1.0 - fraction * (1.0 - (lr_end / lr_peak))


def create_basic_lr_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, lr_peak: float, lr_end: float
) -> LambdaLR:
    """
    Create a simple cosine decay LR scheduler with warmup

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        lr_peak: Peak learning rate
        lr_end: Final learning rate

    Returns:
        LambdaLR scheduler
    """
    decay_steps = max(1, total_steps - warmup_steps)
    lr_function = functools.partial(
        LRScheduleFunctions.cosine_decay_with_warmup, warmup_steps, decay_steps, lr_peak, lr_end
    )

    return LambdaLR(optimizer, lr_lambda=lr_function)


def create_grouped_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule_configs: List[ScheduleConfig],
) -> LambdaLR:
    """
    Create a learning rate scheduler with different schedules for different parameter groups

    Args:
        optimizer: PyTorch optimizer with parameter groups
        schedule_configs: List of schedule configurations for each parameter group

    Returns:
        LambdaLR scheduler with multiple schedules
    """
    # Create a scheduler function for each parameter group
    scheduler_functions = []

    for config in schedule_configs:
        if config.schedule_type == ScheduleType.COSINE:
            decay_steps = max(1, config.total_steps - config.warmup_steps)
            fn = functools.partial(
                LRScheduleFunctions.cosine_decay_with_warmup,
                config.warmup_steps,
                decay_steps,
                config.lr_peak,
                config.lr_end,
            )
        elif config.schedule_type == ScheduleType.LINEAR:
            fn = functools.partial(
                LRScheduleFunctions.linear_decay_with_warmup,
                config.warmup_steps,
                config.total_steps,
                config.lr_peak,
                config.lr_end,
            )
        else:
            raise ValueError(f"Unsupported schedule type: '{config.schedule_type}'")

        scheduler_functions.append(fn)

    return LambdaLR(optimizer, lr_lambda=scheduler_functions)


def get_optimizer_and_scheduler(model: nn.Module, config: Any) -> Tuple[torch.optim.Optimizer, LambdaLR, Any]:
    """
    Set up the optimizer and learning rate scheduler based on configuration

    Args:
        model: PyTorch model
        config: Configuration object with training parameters

    Returns:
        Tuple of (optimizer, scheduler, schedule_config)
    """
    # Standard optimizer for models without SSM layers
    if config.model.ssm_layer == "none":
        optimizer = create_optimizer(model, config.optimizer.lr)
        lr_scheduler = create_basic_lr_scheduler(
            optimizer, config.training.warmup_steps, config.training.steps, config.optimizer.lr, config.optimizer.lr_end
        )

        lr_scheduler_config = ScheduleConfig(
            config.optimizer.lr_schedule,
            config.training.warmup_steps,
            config.training.steps,
            config.optimizer.lr,
            config.optimizer.lr_end,
            "standard",
        )

    # Specialized optimizer for models with SSM layers
    else:
        optimizer, lr_scheduler_configs = create_specialized_optimizer(
            model,
            config.optimizer.lr,
            config.optimizer.lr_ssm,
            config.optimizer.lr_end,
            config.training.warmup_steps,
            config.training.steps,
            config.optimizer.lr_schedule,
            config.optimizer.lr_ssm_schedule,
            config.training.adapter_method,
        )

        lr_scheduler = create_grouped_lr_scheduler(optimizer, lr_scheduler_configs)
        lr_scheduler_config = lr_scheduler_configs

    return optimizer, lr_scheduler, lr_scheduler_config
