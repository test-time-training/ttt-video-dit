import argparse
import json
import os
import os.path as osp
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, get_type_hints


@dataclass
class JobExpConfig:
    """Job-level configs."""

    config_file: Optional[str] = field(default=None, metadata={"help": "Job config file"})
    exp_name: str = field(default="default job", metadata={"help": "Description of the job"})
    dump_folder: str = field(
        default=os.path.join(os.getcwd(), "exp"), metadata={"help": "Location to dump logs of the job"}
    )
    seed: int = field(default=42, metadata={"help": "Random seed for the job"})


@dataclass
class ModelConfig:
    """Model configuration options."""

    name: str = field(default="cogvideo", metadata={"help": "Which model to train", "choices": ["cogvideo"]})
    size: str = field(default="5B", metadata={"help": "Which model size to train"})
    video_length: str = field(default="3sec", metadata={"help": "Which video duration to train"})
    norm_eps: float = field(default=1e-6, metadata={"help": "Eps of layer normalization"})
    scale_factor: float = field(default=1.0, metadata={"help": "Latent scale_factor"})
    ssm_layer: str = field(
        default="ttt_mlp",
        metadata={
            "choices": ["ttt_mlp", "ttt_linear"],
            "help": "Type of sequence modeling block to use [ttt_linear, ttt_mlp]",
        },
    )
    ttt_base_lr: float = field(default=0.1, metadata={"help": "Base learning rate for TTT"})
    mini_batch_size: int = field(default=64, metadata={"help": "Mini batch size for TTT"})


@dataclass
class TrainingConfig:
    """Training configuration options."""

    adapter_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "The fine-tuning method to use: 'sft' for full fine-tuning with parameter groups, 'qkvo' for query, key, value, output fine-tuning.",
            "choices": ["sft", "qkvo"],
        },
    )
    dataset_path: Optional[str] = field(default=None, metadata={"help": "Path to the dataset in the file system."})
    jsonl_paths: Optional[str] = field(default=None, metadata={"help": "Jsonl path for preembedding dataset."})
    global_batch_size: int = field(default=8, metadata={"help": "Global batch size."})
    grad_accum_steps: int = field(default=1, metadata={"help": "Grad accumulatation steps."})
    warmup_steps: int = field(default=50, metadata={"help": "The number of steps for lr scheduler warmup."})
    steps: int = field(default=5000, metadata={"help": "How many train steps to run"})
    gc_freq: int = field(default=50, metadata={"help": "Python garbage control scheduling interval, in steps"})


@dataclass
class EvalConfig:
    """Evaluation-specific configuration options."""

    input_file: Optional[str] = field(default=None, metadata={"help": "Path to a jsonl file with prompts"})
    output_dir: str = field(default="./output", metadata={"help": "Directory to save generated results"})

    image_width: int = field(default=720, metadata={"help": "Width of the generated image"})
    image_height: int = field(default=480, metadata={"help": "Height of the generated image"})
    sampling_fps: int = field(default=16, metadata={"help": "Frames per second of generated video"})
    sampling_num_frames: int = field(default=13, metadata={"help": "Number of frames to sample"})
    latent_channels: int = field(default=16, metadata={"help": "Number of channels in latent space"})

    num_denoising_steps: int = field(default=50, metadata={"help": "Number of denoising steps"})
    scale_factor: float = field(default=0.7, metadata={"help": "Scale factor for sampling"})
    dtype: str = field(default="bfloat16", metadata={"help": "Datatype for sampling: [bfloat16, float16, float32]"})

    vae_checkpoint_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the VAE checkpoint for decoding"}
    )
    vae_scale_factor: float = field(default=1.0, metadata={"help": "Scale factor used during VAE decoding"})

    txt_maxlen: int = field(default=498, metadata={"help": "Maximum token length for T5 input"})
    t5_model_dir: Optional[str] = field(default=None, metadata={"help": "Directory path to the T5 model"})


@dataclass
class GuiderConfig:
    """Classifier guider-specific configuration options."""

    scale: int = field(default=6, metadata={"help": "Scale factor for the classifier guider"})
    exp: int = field(default=5, metadata={"help": "Exponent for the classifier guider"})
    num_steps: int = field(default=50, metadata={"help": "Number of steps for the classifier guider"})


@dataclass
class DenoiserConfig:
    """Sampling denoiser-specific configuration options."""

    num_idx: int = field(default=1000, metadata={"help": "Number of indices for the denoiser"})
    quantize_c_noise: bool = field(default=False, metadata={"help": "Quantize c noise for the denoiser"})


@dataclass
class DiscretizationConfig:
    """Sampling discretization-specific configuration options."""

    shift_scale: float = field(default=1.0, metadata={"help": "Shift scale for the discretization"})


@dataclass
class OptimizerConfig:
    """Optimizer configuration options."""

    name: str = field(default="AdamW", metadata={"help": "Which optimizer to use", "choices": ["AdamW"]})
    lr: float = field(default=1e-4, metadata={"help": "Learning rate to use on all parameters outside of the ssm"})
    lr_end: float = field(
        default=0.0,
        metadata={
            "help": "End learning rate to use on all parameters. Controls the schedule for all parameter groups."
        },
    )
    lr_ssm: float = field(default=1e-4, metadata={"help": "Learning rate to use for the ssm"})
    lr_schedule: str = field(default="linear", metadata={"help": "Learning rate schedule to use. [cosine, linear]"})
    lr_ssm_schedule: str = field(
        default="linear", metadata={"help": "Learning rate schedule to use for the ssm. [cosine, linear]"}
    )
    gradient_clipping_norm: float = field(default=0.1, metadata={"help": "Norm for gradient clipping"})


@dataclass
class CheckpointConfig:
    """Checkpoint configuration options."""

    init_state_dir: Optional[str] = field(default=None, metadata={"help": "Path to the model weights."})
    interval: int = field(default=0, metadata={"help": "Interval at which to save checkpoints."})
    resume: bool = field(default=False, metadata={"help": "Resume experiment.", "action": "store_true"})
    resume_step: int = field(
        default=-1, metadata={"help": "At which step to resume from checkpoint. Use -1 for auto-detect latest."}
    )
    timeout_minutes: int = field(
        default=0, metadata={"help": "Duration before the job times out, used for checkpointing before timeout."}
    )


@dataclass
class ParallelismConfig:
    """Parallelism configuration options."""

    fsdp_unsharded_dtype: str = field(
        default="bfloat16",
        metadata={
            "help": "Dtype that model computations will be run after FSDP gather [float32, bfloat16, float16]",
            "choices": ["float32", "bfloat16", "float16"],
        },
    )
    tp_sharding: int = field(
        default=1, metadata={"help": "The number of gpus to shard the tensor parallelism. Use 1 for no sharding"}
    )
    dp_sharding: int = field(
        default=8, metadata={"help": "The number of gpus to shard for data parallelism. Use 1 for no sharding"}
    )
    dp_replicate: int = field(
        default=1, metadata={"help": "The number of times to replicate the data parallelism. Use 1 for no replication"}
    )


@dataclass
class RematConfig:
    """Remat configuration options."""

    transformer_checkpoint_layer_group_size: int = field(
        default=1, metadata={"help": "Number of transformer layers to group together for remat."}
    )
    scan_checkpoint_group_size: int = field(default=16, metadata={"help": "Scan checkpoint group size for TTT"})
    forward_ssm: bool = field(
        default=False, metadata={"help": "Apply activation checkpointing to forward ssm block.", "action": "store_true"}
    )
    reverse_ssm: bool = field(
        default=False, metadata={"help": "Apply activation checkpointing to reverse ssm block.", "action": "store_true"}
    )
    attention: bool = field(
        default=False, metadata={"help": "Apply activation checkpointing to attention block.", "action": "store_true"}
    )
    mlp: bool = field(
        default=False, metadata={"help": "Apply activation checkpointing to mlp block.", "action": "store_true"}
    )
    seq_modeling_block: bool = field(
        default=False,
        metadata={"help": "Apply activation checkpointing to sequential modeling block.", "action": "store_true"},
    )
    shard_transformer_inputs: bool = field(
        default=False,
        metadata={
            "help": "Sharde transformer inputs across the tp mesh for remat. Requires tensor parallelism.",
            "action": "store_true",
        },
    )


@dataclass
class CommConfig:
    """Communication configuration options."""

    init_timeout_seconds: int = field(
        default=1200,
        metadata={"help": "Timeout for communication operations, during initialization and first train step."},
    )


@dataclass
class WandBConfig:
    """Weights & Biases configuration options."""

    disable: bool = field(default=False, metadata={"help": "Disable WandB logging", "action": "store_true"})
    project: str = field(default="ttt-video", metadata={"help": "WandB project name"})
    entity: str = field(default="default", metadata={"help": "WandB entity name"})
    log_interval: int = field(default=50, metadata={"help": "WandB log interval"})
    alert: bool = field(
        default=False, metadata={"help": "Send a notification when a prompt is processed", "action": "store_true"}
    )


def string_list(input_str: str) -> List[str]:
    """Parse a comma-separated string into a list of strings."""
    return input_str.split(",")


class JobConfig:
    """
    A helper class to manage the train configuration.
    Semantics:
    - Default config is loaded from a toml file. If no toml file is provided,
    then the default config is loaded from argparse defaults.
    - if toml file has missing keys, they are filled with argparse defaults.
    - if additional explicit cmd args are provided in addition to the toml
    file, they will override the toml config and the argparse defaults

    precedence order: cmdline > toml > argparse default

    Arg parsing semantics:

    Each argument starts with <prefix>_ which is the config name in the toml file
    followed by name of the option in the toml file. For ex,
    model.name translates to:
        [model]
        name
    in the toml file
    """

    # Define class-level attributes for editor intellisense
    job: JobExpConfig
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    checkpoint: CheckpointConfig
    parallelism: ParallelismConfig
    remat: RematConfig
    comm: CommConfig
    wandb: WandBConfig
    eval: EvalConfig
    guider: GuiderConfig
    denoiser: DenoiserConfig
    discretization: DiscretizationConfig

    # Map config names to their corresponding dataclass types
    _config_types: Dict[str, Type] = {
        "job": JobExpConfig,
        "model": ModelConfig,
        "training": TrainingConfig,
        "optimizer": OptimizerConfig,
        "checkpoint": CheckpointConfig,
        "parallelism": ParallelismConfig,
        "remat": RematConfig,
        "comm": CommConfig,
        "wandb": WandBConfig,
    }

    def __init__(self, eval_mode=False):
        # Initialize config attributes with default values
        for config_name, config_class in self._config_types.items():
            setattr(self, config_name, config_class())

        # Initialize empty config map
        self.config_map = None

        # Create main parser
        self.parser = argparse.ArgumentParser(description="ttt arg parser.")

        if eval_mode:
            self._setup_eval_args()

        # Auto-generate argparse arguments from dataclass fields
        self._generate_args_from_dataclasses()

    def _setup_eval_args(self):
        """Add eval config to the config types."""
        self._config_types["eval"] = EvalConfig
        self._config_types["guider"] = GuiderConfig
        self._config_types["denoiser"] = DenoiserConfig
        self._config_types["discretization"] = DiscretizationConfig

    def _generate_args_from_dataclasses(self):
        """Generate argparse arguments automatically from dataclass fields and their metadata."""
        for config_name, config_class in self._config_types.items():
            # Get all fields in the dataclass
            fields = config_class.__dataclass_fields__
            type_hints = get_type_hints(config_class)

            for field_name, field_info in fields.items():
                # Get the type and metadata
                field_type = type_hints.get(field_name)
                metadata = field_info.metadata
                default = field_info.default

                # Build argument name
                arg_name = f"--{config_name}.{field_name}"

                # Get help text and other metadata
                help_text = metadata.get("help", "")
                choices = metadata.get("choices", None)
                action = metadata.get("action", None)

                # Add the argument to the parser
                kwargs = {"help": help_text}

                # Handle different argument types
                if action:
                    # For boolean flags with store_true
                    kwargs["action"] = action
                    # Don't set default for store_true/store_false actions
                    if action not in ["store_true", "store_false"]:
                        kwargs["default"] = default
                else:
                    # For regular arguments
                    if field_name == "experimental.pipeline_parallel_split_points":
                        kwargs["type"] = string_list
                    elif field_type == Optional[str] and default is None:
                        kwargs["type"] = str
                        kwargs["default"] = None
                    elif field_type == Optional[int] and default is None:
                        kwargs["type"] = int
                        kwargs["default"] = None
                    else:
                        # Use the appropriate type
                        kwargs["type"] = type(default) if default is not None else field_type
                        kwargs["default"] = default

                    # Add choices if specified
                    if choices:
                        kwargs["choices"] = choices

                self.parser.add_argument(arg_name, **kwargs)

    def parse_args(self, args_list: list = sys.argv[1:]):
        args, cmd_args = self.parse_args_from_command_line(args_list)
        config_file = getattr(args, "job.config_file", None)
        # build up a two level dict
        args_dict = self._args_to_two_level_dict(args)
        if config_file is not None:
            try:
                with open(config_file, "rb") as f:
                    for k, v in tomllib.load(f).items():
                        # to prevent overwrite of non-specified keys
                        args_dict[k] |= v
            except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
                # logger.exception(f"Error while loading the configuration file: {config_file}")
                # logger.exception(f"Error details: {str(e)}")
                raise e

        # override args dict with cmd_args
        cmd_args_dict = self._args_to_two_level_dict(cmd_args)
        for section, section_args in cmd_args_dict.items():
            for k, v in section_args.items():
                args_dict[section][k] = v

        # Save for logging config later
        self.config_map = args_dict

        # Update the instance attributes with parsed values
        for config_name, config_values in args_dict.items():
            if config_name in self._config_types:
                config_class = self._config_types[config_name]
                # Check for unexpected keyword arguments
                valid_fields = set(config_class.__dataclass_fields__.keys())
                provided_fields = set(config_values.keys())
                unexpected_fields = provided_fields - valid_fields

                if unexpected_fields:
                    raise TypeError(
                        f"Invalid field(s) found in {config_name} configuration: {', '.join(unexpected_fields)}.\n"
                        f"Valid fields for {config_name} are: {', '.join(sorted(valid_fields))}"
                    )

                # Create a new instance of the config dataclass with updated values
                config_instance = config_class(**config_values)
                setattr(self, config_name, config_instance)

        self._validate_config()

    def _args_to_two_level_dict(self, args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
        args_dict = defaultdict(dict)
        for k, v in vars(args).items():
            first_level_key, second_level_key = k.split(".", 1)
            args_dict[first_level_key][second_level_key] = v
        return args_dict

    def _validate_config(self) -> None:
        assert self.model.name
        assert self.model.size

        assert osp.isabs(self.job.dump_folder), "dump_folder should be an absolute path"

        if self.remat.shard_transformer_inputs:
            assert self.parallelism.tp_sharding > 1, "Sharding transformer inputs requires tensor parallelism"

    def parse_args_from_command_line(self, args_list) -> Tuple[argparse.Namespace, argparse.Namespace]:
        """
        Parse command line arguments and return the parsed args and the command line only args
        """
        args = self.parser.parse_args(args_list)

        # aux parser to parse the command line only args, with no defaults from main parser
        aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        for arg, val in vars(args).items():
            if isinstance(val, bool):
                aux_parser.add_argument("--" + arg, action="store_true" if val else "store_false")
            elif arg == "experimental.pipeline_parallel_split_points":
                # without this special case, type inference breaks here,
                # since the inferred type is just 'list' and it ends up flattening
                # e.g. from ["layers.0", "layers.1"] into ["l", "a", "y", "e", "r", "s", ".0", ...]
                aux_parser.add_argument("--" + arg, type=string_list)
            else:
                aux_parser.add_argument("--" + arg, type=type(val))

        cmd_args, _ = aux_parser.parse_known_args(args_list)

        return args, cmd_args

    def to_dict(self) -> dict:
        """
        Returns the args of the instance as a nested dictionary.
        """
        assert self.config_map is not None, "Attempting to retrieve dict of args before parsing args."

        return self.config_map

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4)
