"""
Reading runtime parameters from a standard configuration file (which is easier
to work on than JSON).
"""

import argparse
import logging
import os
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import IO, Any

import torch
import yaml


class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self.root_path = os.path.split(stream.name)[0]
        except AttributeError:
            self.root_path = os.path.curdir

        super().__init__(stream)


class Config:
    """
    Retrieving configuration parameters by parsing a configuration file
    using the YAML configuration file parser.
    """

    _instance = None

    @staticmethod
    def construct_include(loader: Loader, node: yaml.Node) -> Any:
        """Include file referenced at node."""
        with open(
            Path(loader.name)
            .parent.joinpath(loader.construct_yaml_str(node))
            .resolve(),
            "r",
        ) as f:
            return yaml.load(f, type(loader))

    def __new__(cls):
        if cls._instance is None:
            parser = argparse.ArgumentParser()

            parser.add_argument(
                "-u", "--cpu", action="store_true", help="Use CPU as the device."
            )
            parser.add_argument(
                "-m", "--mps", action="store_true", help="Use MPS as the device."
            )
            parser.add_argument(
                "-d", "--debug", action="store_true", help="Debug mode."
            )
            parser.add_argument(
                "--no_tracemalloc",
                default=False,
                action="store_true",
                help="Flag to stop memory allocation tracing during training. This could speed up training on Windows.",
            )
            parser.add_argument(
                "--push_to_hub",
                action="store_true",
                help="Whether or not to push the model to the Hub.",
            )
            parser.add_argument(
                "--hub_token",
                type=str,
                default=None,
                help="The token to use to push to the Model Hub.",
            )
            parser.add_argument(
                "--hub_model_id",
                type=str,
                default=None,
                help="The name of the repository to keep in sync with the local `output_dir`.",
            )
            parser.add_argument(
                "--logging_dir",
                type=str,
                default="logs",
                help=(
                    "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
                    " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
                ),
            )
            parser.add_argument(
                "--allow_tf32",
                action="store_true",
                help=(
                    "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                    " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
                ),
            )
            parser.add_argument(
                "--report_to",
                type=str,
                default="tensorboard",
                help=(
                    'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                    ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
                ),
            )
            parser.add_argument(
                "--wandb_key",
                type=str,
                default=None,
                help=(
                    "If report to option is set to wandb, api-key for wandb used for login to wandb "
                ),
            )
            parser.add_argument(
                "--wandb_project_name",
                type=str,
                default=None,
                help=(
                    "If report to option is set to wandb, project name in wandb for log tracking  "
                ),
            )
            parser.add_argument(
                "--mixed_precision",
                type=str,
                default=None,
                choices=["no", "fp16", "bf16"],
                help=(
                    "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                    " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                    " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
                ),
            )
            parser.add_argument(
                "--prior_generation_precision",
                type=str,
                default=None,
                choices=["no", "fp32", "fp16", "bf16"],
                help=(
                    "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                    " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
                ),
            )
            parser.add_argument(
                "--local_rank",
                type=int,
                default=-1,
                help="For distributed training: local_rank",
            )
            parser.add_argument(
                "--enable_xformers_memory_efficient_attention",
                action="store_true",
                help="Whether or not to use xformers.",
            )
            parser.add_argument(
                "-l", "--log", type=str, default="info", help="Log messages level."
            )
            parser.add_argument(
                "-c",
                "--config",
                type=str,
                default="./config.yml",
                help="Configuration file.",
            )
            parser.add_argument(
                "-r",
                "--resume",
                action="store_true",
                help="Resume a previously interrupted training session.",
            )
            args = parser.parse_args()

            env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
            if env_local_rank != -1 and env_local_rank != args.local_rank:
                args.local_rank = env_local_rank
            Config.args = args

            numeric_level = getattr(logging, args.log.upper(), None)

            if not isinstance(numeric_level, int):
                raise ValueError(f"Invalid log level: {args.log}")

            logging.basicConfig(
                filename=f"inversion_network_{os.getpid()}.log",
                format="[%(levelname)s][%(asctime)s]: %(message)s",
                datefmt="%H:%M:%S",
            )

            root_logger = logging.getLogger()
            root_logger.setLevel(numeric_level)

            cls._instance = super(Config, cls).__new__(cls)

            if "config_file" in os.environ:
                filename = os.environ["config_file"]
            else:
                filename = args.config

            yaml.add_constructor("!include", Config.construct_include, Loader)

            if os.path.isfile(filename):
                with open(filename, "r", encoding="utf-8") as config_file:
                    config = yaml.load(config_file, Loader)
            else:
                # if the configuration file does not exist, raise an error
                raise ValueError("A configuration file must be supplied.")

            Config.basic = Config.namedtuple_from_dict(config["basic"])
            Config.training = Config.namedtuple_from_dict(config["training"])
            Config.model = Config.namedtuple_from_dict(config["model"])
        return cls._instance

    @staticmethod
    def namedtuple_from_dict(obj):
        """Creates a named tuple from a dictionary."""
        if isinstance(obj, dict):
            fields = sorted(obj.keys())
            namedtuple_type = namedtuple(
                typename="Config", field_names=fields, rename=True
            )
            field_value_pairs = OrderedDict(
                (str(field), Config.namedtuple_from_dict(obj[field]))
                for field in fields
            )
            try:
                return namedtuple_type(**field_value_pairs)
            except TypeError:
                # Cannot create namedtuple instance so fallback to dict (invalid attribute names)
                return dict(**field_value_pairs)
        elif isinstance(obj, (list, set, tuple, frozenset)):
            return [Config.namedtuple_from_dict(item) for item in obj]
        else:
            return obj

    @staticmethod
    def gpu_count() -> int:
        """Returns the number of GPUs available for training."""
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        elif Config.args.mps and torch.backends.mps.is_built():
            return 1
        else:
            return 0

    @staticmethod
    def device() -> str:
        """Returns the device to be used for training."""
        device = "cpu"

        if Config.args.cpu:
            return device
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            if Config.gpu_count() > 1 and isinstance(Config.args.id, int):
                # A client will always run on the same GPU
                gpu_id = Config.args.id % torch.cuda.device_count()
                device = f"cuda:{gpu_id}"
            else:
                device = "cuda:0"

        if Config.args.mps and torch.backends.mps.is_built():
            device = "mps"
        return device
