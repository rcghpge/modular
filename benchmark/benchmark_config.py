# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Benchmark configuration classes with inheritance structure for MAX benchmarks."""

import argparse
import enum
import logging
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    from .benchmark_datasets import (  # type: ignore[import-not-found, unused-ignore]
        DATASET_REGISTRY,
        DatasetMode,
    )
except ImportError:
    from benchmark_datasets import (  # type: ignore[import-not-found, unused-ignore, no-redef]
        DATASET_REGISTRY,
        DatasetMode,
    )

logger = logging.getLogger("max.benchmark")

# Workaround for when we don't have max.pipelines installed. This assumes that
# we copied the max_config.py file to the current directory.
try:
    from max.pipelines.lib import (  # type: ignore[import-not-found, unused-ignore, no-redef]
        MAXConfig,
        deep_merge_max_configs,
    )
except (ImportError, ModuleNotFoundError):
    logger.warning(
        "max.pipelines.lib not found, using max_config.py from current directory"
    )
    # Also type: ignore because we don't want mypy to trigger on this since
    # it's intentional anyway.
    from max_config import MAXConfig, deep_merge_max_configs  # type: ignore


class Backend(str, enum.Enum):
    vllm = "vllm"
    vllm_chat = "vllm-chat"
    trt_llm = "trt-llm"
    modular = "modular"
    modular_chat = "modular-chat"
    sglang = "sglang"
    sglang_chat = "sglang-chat"


class Endpoint(str, enum.Enum):
    completions = "/v1/completions"
    chat_completions = "/v1/chat/completions"
    ensemble_generate_stream = "/v2/models/ensemble/generate_stream"


def _add_config_file_arg_to_parser(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add the --config-file argument to a parser.

    Args:
        parser: The parser to add the argument to.
    """
    parser.add_argument(
        "--config-file",
        type=Path,
        help="Path to configuration file. If provided, this config will inherit from the default config and override its values.",
    )
    return parser


def _resolve_user_provided_config_file_cli_arg(
    args: Optional[Sequence[str]] = None,
) -> tuple[Optional[Path], list[str]]:
    """Resolve the user-provided --config-file argument from command line arguments.

    This utility function extracts the config file path from command line arguments
    before the main argument parsing, allowing the config file to be loaded and used
    as defaults for the main parser.

    Args:
        args: Command line arguments to parse. If None, parse from sys.argv.

    Returns:
        Tuple of (config_file_path, remaining_args) where:
        - config_file_path: Path to the config file if provided, None otherwise
        - remaining_args: List of remaining arguments after removing --config-file
    """
    # Create a preliminary parser to get the config file path
    preliminary_parser = argparse.ArgumentParser(add_help=False)
    preliminary_parser = _add_config_file_arg_to_parser(preliminary_parser)

    # Parse preliminary args to get config file path
    preliminary_args, remaining_args = preliminary_parser.parse_known_args(
        args=args
    )
    return preliminary_args.config_file, remaining_args


@dataclass
class BaseBenchmarkConfig(MAXConfig):
    """Base configuration class containing parameters common to all benchmark types.

    This class contains the core parameters that are shared across all benchmark types:
    - Model and tokenizer configuration
    - Basic dataset configuration
    - Common workload parameters
    - Basic output control
    - Result saving configuration
    - Common control flags
    """

    # Config file section name for MAXConfig interface
    _config_file_section_name: str = "benchmark_config"
    """The section name to use when loading this config from a MAXConfig file."""

    # Model and tokenizer configuration (common to all benchmarks)
    model: Optional[str] = None
    """Name of the model. Required when running benchmark."""

    tokenizer: Optional[str] = None
    """Name or path of the tokenizer, if not using the default tokenizer."""

    model_max_length: Optional[int] = None
    """Override for tokenizer max length. Needed if server has a lower max length than the tokenizer."""

    trust_remote_code: bool = False
    """Trust remote code from huggingface."""

    # Dataset configuration (common across all benchmark types)
    dataset_name: str = "sharegpt"
    """Name of the dataset to benchmark on."""

    dataset_path: Optional[str] = None
    """Path to the dataset."""

    dataset_mode: DatasetMode = DatasetMode.HUGGINGFACE
    """Mode for loading the dataset: LOCAL (from local path/env var) or HUGGINGFACE (HuggingFace Hub)."""

    # Basic workload parameters
    num_prompts: int = 1000
    """Number of prompts to process."""

    seed: int = 0
    """Random seed for reproducibility."""

    # Control flags
    disable_tqdm: bool = False
    """Specify to disable tqdm progress bar."""

    print_inputs_and_outputs: bool = False
    """Print all input and outputs to console."""

    # Unknown fields storage (not a dataclass field)
    _unknown_fields: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )

    @staticmethod
    def help() -> dict[str, str]:
        """Documentation for base benchmark config parameters.

        Returns:
            Dictionary of config options and their descriptions.
        """
        return {
            "model": "Name of the model. Required when running benchmark.",
            "tokenizer": "Name or path of the tokenizer, if not using the default tokenizer.",
            "trust_remote_code": "Trust remote code from huggingface.",
            "dataset_name": "Name of the dataset to benchmark on.",
            "dataset_path": "Path to the dataset.",
            "dataset_mode": "Mode for loading the dataset: LOCAL (from local path/env var) or HUGGINGFACE (HuggingFace Hub).",
            "num_prompts": "Number of prompts to process.",
            "seed": "Random seed for reproducibility.",
            "disable_tqdm": "Specify to disable tqdm progress bar.",
            "print_inputs_and_outputs": "Print all input and outputs to console.",
        }

    @staticmethod
    def get_default_field_choices() -> dict[str, list[str]]:
        """Get valid choices for fields that have constrained values.

        Returns:
            Dictionary mapping field names to their valid choices.
        """
        return {
            # TODO: Propagate proper enum choices here than just the string values
            "backend": [backend.value for backend in Backend],
            "endpoint": [endpoint.value for endpoint in Endpoint],
            "dataset_name": list(DATASET_REGISTRY.keys()),
            "dataset_mode": [mode.value for mode in DatasetMode],
            "random_distribution_type": ["uniform", "normal"],
        }

    @classmethod
    def get_default_required_fields(cls) -> set[str]:
        """Get required fields for the benchmark config."""
        return super().get_default_required_fields().union({"model"})


@dataclass
class ServingBenchmarkConfig(BaseBenchmarkConfig):
    """Configuration class for serving benchmarks (benchmark_serving.py).

    Inherits from BaseBenchmarkConfig and adds serving-specific parameters:
    - Backend and API configuration
    - Request configuration (concurrency, LoRA)
    - Traffic control (request rate, burstiness, TTFT)
    - Chat session configuration
    - Serving-specific dataset parameters
    - GPU stats collection
    """

    # Backend and API configuration (serving-specific)
    # TODO: Propagate proper enum choices here than just the string values
    backend: str = field(
        default=Backend.modular.value,
        metadata={
            "group": "Backend and API Configuration",
            "group_description": "Configuration for backend selection and API endpoints",
        },
    )
    """Backend to use for benchmarking. Choices: vllm, vllm-chat, trt-llm, modular, modular-chat, sglang, sglang-chat"""

    base_url: Optional[str] = field(
        default=None, metadata={"group": "Backend and API Configuration"}
    )
    """Server or API base url if not using http host and port."""

    host: str = field(
        default="localhost", metadata={"group": "Backend and API Configuration"}
    )
    """Server host."""

    port: int = field(
        default=8000, metadata={"group": "Backend and API Configuration"}
    )
    """Server port."""

    endpoint: str = field(
        default=Endpoint.chat_completions.value,
        metadata={"group": "Backend and API Configuration"},
    )
    """API endpoint. Choices: /v1/completions, /v1/chat/completions, /v2/models/ensemble/generate_stream"""

    # Request configuration (serving-specific)
    max_concurrency: Optional[int] = field(
        default=None,
        metadata={
            "group": "Request Configuration",
            "group_description": "Parameters controlling request concurrency and processing",
        },
    )
    """Maximum concurrent requests (optimized for serving benchmarks)."""

    lora: Optional[str] = field(
        default=None, metadata={"group": "Request Configuration"}
    )
    """Optional LoRA name."""

    # Workload configuration (serving-specific)
    max_benchmark_duration_s: Optional[int] = field(
        default=None,
        metadata={
            "group": "Workload Configuration",
            "group_description": "Parameters controlling benchmark duration and workload characteristics",
        },
    )
    """Maximum benchmark duration in seconds."""

    num_chat_sessions: Optional[int] = field(
        default=None, metadata={"group": "Workload Configuration"}
    )
    """Number of multiturn chat sessions."""

    delay_between_chat_turns: Optional[int] = field(
        default=None, metadata={"group": "Workload Configuration"}
    )
    """Delay between chat turns in ms."""

    # Output control (serving-specific extensions)
    output_lengths: Optional[str] = field(
        default=None,
        metadata={
            "group": "Output Control",
            "group_description": "Parameters controlling output generation and sampling",
        },
    )
    """Path to YAML file with output lengths or int."""

    max_output_len: Optional[int] = field(
        default=None, metadata={"group": "Output Control"}
    )
    """Maximum output length per request."""

    temperature: float = field(
        default=0.0, metadata={"group": "Output Control"}
    )
    """Temperature for sampling."""

    top_p: float = field(default=1.0, metadata={"group": "Output Control"})
    """Top-p for sampling."""

    top_k: Optional[int] = field(
        default=None, metadata={"group": "Output Control"}
    )
    """Top-k for sampling."""

    # Traffic control (serving-specific)
    request_rate: float = field(
        default=float("inf"),
        metadata={
            "group": "Traffic Control",
            "group_description": "Parameters controlling request rate and traffic patterns",
        },
    )
    """Requests per second (finite rate for realistic benchmarking)."""

    burstiness: float = field(
        default=1.0, metadata={"group": "Traffic Control"}
    )
    """Burstiness factor (1.0 = Poisson process)."""

    skip_first_n_requests: int = field(
        default=0, metadata={"group": "Traffic Control"}
    )
    """Skip first N requests for measurements."""

    chat_warmup_delay_ms: float = field(
        default=0.0, metadata={"group": "Traffic Control"}
    )
    """Delay between starting chat sessions."""

    ignore_first_turn_stats: bool = field(
        default=False, metadata={"group": "Traffic Control"}
    )
    """Ignore the first turn statistics in multiturn chat sessions."""

    # Dataset-specific parameters (serving workloads)
    arxiv_summarization_input_len: int = field(
        default=15000,
        metadata={
            "group": "Dataset-Specific Parameters",
            "group_description": "Parameters specific to different dataset types and workloads",
        },
    )
    obfuscated_conversations_average_output_len: int = field(
        default=175, metadata={"group": "Dataset-Specific Parameters"}
    )
    obfuscated_conversations_coefficient_of_variation: float = field(
        default=0.1, metadata={"group": "Dataset-Specific Parameters"}
    )
    obfuscated_conversations_shuffle: bool = field(
        default=False, metadata={"group": "Dataset-Specific Parameters"}
    )
    random_coefficient_of_variation: str = field(
        default="0.3,0.7", metadata={"group": "Dataset-Specific Parameters"}
    )
    random_distribution_type: str = field(
        default="normal",  # choices: uniform, normal
        metadata={"group": "Dataset-Specific Parameters"},
    )
    random_first_turn_ratio: float = field(
        default=1.0, metadata={"group": "Dataset-Specific Parameters"}
    )
    random_image_size: str = field(
        default="", metadata={"group": "Dataset-Specific Parameters"}
    )
    random_input_len: int = field(
        default=1024, metadata={"group": "Dataset-Specific Parameters"}
    )
    random_max_num_unique_sys_prompt: int = field(
        default=1, metadata={"group": "Dataset-Specific Parameters"}
    )
    random_num_turns: int = field(
        default=1, metadata={"group": "Dataset-Specific Parameters"}
    )
    random_output_len: int = field(
        default=128, metadata={"group": "Dataset-Specific Parameters"}
    )
    random_sys_prompt_ratio: float = field(
        default=0.0, metadata={"group": "Dataset-Specific Parameters"}
    )
    sonnet_input_len: int = field(
        default=550, metadata={"group": "Dataset-Specific Parameters"}
    )
    sonnet_prefix_len: int = field(
        default=200, metadata={"group": "Dataset-Specific Parameters"}
    )

    # Control flags (serving-specific)
    skip_test_prompt: bool = field(
        default=False,
        metadata={
            "group": "Control Flags",
            "group_description": "Boolean flags controlling benchmark behavior",
        },
    )
    collect_gpu_stats: bool = field(
        default=False, metadata={"group": "Control Flags"}
    )
    """Enable GPU stats collection for serving benchmarks."""

    collect_cpu_stats: bool = field(
        default=True, metadata={"group": "Control Flags"}
    )
    """Enable CPU stats collection for serving benchmarks."""

    # Result saving (serving-specific extensions)
    server_args: str = field(
        default="",
        metadata={
            "group": "Result Saving",
            "group_description": "Parameters controlling result saving and output",
        },
    )
    """Server arguments string."""

    # Result saving (serving-specific extensions)
    save_result: bool = field(
        default=False, metadata={"group": "Result Saving"}
    )
    """Specify to save benchmark results to a json file."""

    record_output_lengths: Optional[str] = field(
        default=None, metadata={"group": "Result Saving"}
    )
    """Path to save output lengths in YAML format."""

    result_dir: Optional[str] = field(
        default=None, metadata={"group": "Result Saving"}
    )
    """Directory to save results."""

    result_filename: Optional[str] = field(
        default=None, metadata={"group": "Result Saving"}
    )
    """Custom filename (auto-generated if null)."""

    metadata: list[str] = field(
        default_factory=list, metadata={"group": "Result Saving"}
    )
    """Key-value pairs for metadata (format: ["key=value", ...])."""

    @staticmethod
    def help() -> dict[str, str]:
        """Documentation for serving benchmark config parameters.

        Returns:
            Dictionary of config options and their descriptions.
        """
        # Get base help and extend with serving-specific parameters
        base_help = BaseBenchmarkConfig.help()
        serving_help = {
            "backend": "Backend to use for benchmarking. Choices: vllm, vllm-chat, trt-llm, modular, modular-chat, sglang, sglang-chat",
            "base_url": "Server or API base url if not using http host and port.",
            "host": "Server host.",
            "port": "Server port.",
            "endpoint": "API endpoint. Choices: /v1/completions, /v1/chat/completions, /v2/models/ensemble/generate_stream",
            "max_concurrency": "Maximum concurrent requests (optimized for serving benchmarks).",
            "lora": "Optional LoRA name.",
            "max_benchmark_duration_s": "Maximum benchmark duration in seconds.",
            "num_chat_sessions": "Number of multiturn chat sessions.",
            "delay_between_chat_turns": "Delay between chat turns in ms.",
            "output_lengths": "Path to YAML file with output lengths or int.",
            "max_output_len": "Maximum output length per request.",
            "temperature": "Temperature for sampling.",
            "top_p": "Top-p for sampling.",
            "request_rate": "Requests per second (finite rate for realistic benchmarking).",
            "burstiness": "Burstiness factor (1.0 = Poisson process).",
            "skip_first_n_requests": "Skip first N requests for measurements.",
            "chat_warmup_delay_ms": "Delay between starting chat sessions.",
            "sonnet_input_len": "Number of input tokens per request, used only for sonnet dataset.",
            "sonnet_prefix_len": "Number of prefix tokens per request, used only for sonnet dataset.",
            "arxiv_summarization_input_len": "Number of input tokens per request, used only for arxiv-summarization dataset.",
            "obfuscated_conversations_average_output_len": "Average output length for obfuscated-conversations dataset when output_lengths is not provided.",
            "obfuscated_conversations_coefficient_of_variation": "Coefficient of variation for output length for obfuscated-conversations dataset when output_lengths is not provided.",
            "obfuscated_conversations_shuffle": "Shuffle the obfuscated-conversations dataset.",
            "random_coefficient_of_variation": "Coefficient of variation for input/output length, used only for random sampling.",
            "random_distribution_type": "Type of probability distribution for sampled input/output length. Choices: uniform, normal",
            "random_first_turn_ratio": "Ratio of the length of the first turn to the length of subsequent turns.",
            "random_image_size": "Size of random images to generate.",
            "random_input_len": "Number of input tokens per request, used only for random sampling.",
            "random_max_num_unique_sys_prompt": "Maximum number of unique system prompts, used only for random sampling.",
            "random_num_turns": "Number of turns per session, used only for random sampling and --num-chat-sessions.",
            "random_output_len": "Number of output tokens per request, used only for random sampling.",
            "random_sys_prompt_ratio": "Ratio to determine the system prompt length, used only for random sampling.",
            "skip_test_prompt": "Skip the test prompt. Useful when doing external profiling.",
            "collect_gpu_stats": "Enable GPU stats collection for serving benchmarks.",
            "collect_cpu_stats": "Enable CPU stats collection for serving benchmarks.",
            "server_args": "Server arguments string.",
            "save_result": "Specify to save benchmark results to a json file.",
            "result_dir": "Directory to save results.",
            "result_filename": "Custom filename (auto-generated if null).",
            "record_output_lengths": "Path to save output lengths in YAML format.",
            "metadata": 'Key-value pairs for metadata (format: ["key=value", ...]).',
        }
        return {**base_help, **serving_help}

    @classmethod
    def get_default_required_fields(cls) -> set[str]:
        """Get required fields for the benchmark config."""
        return super().get_default_required_fields().union({"dataset_name"})


# Convenience functions for loading specific configuration types
def load_base_benchmark_config(
    config_file: str = "base_config.yaml",
    overrides: Optional[dict[str, Any]] = None,
) -> BaseBenchmarkConfig:
    """Load base benchmark configuration with optional overrides.

    Args:
        config_file: Path to configuration file
        overrides: Optional dictionary of parameter overrides

    Returns:
        BaseBenchmarkConfig instance
    """
    config = BaseBenchmarkConfig.from_config_file(config_file)
    if overrides:
        for key, value in overrides.items():
            setattr(config, key, value)
    return config


def load_serving_benchmark_config(
    config_file: str = "serving_config.yaml",
    overrides: Optional[dict[str, Any]] = None,
) -> ServingBenchmarkConfig:
    """Load serving benchmark configuration with optional overrides.

    Args:
        config_file: Path to configuration file
        overrides: Optional dictionary of parameter overrides

    Returns:
        ServingBenchmarkConfig instance
    """
    config = ServingBenchmarkConfig.from_config_file(config_file)
    if overrides:
        for key, value in overrides.items():
            setattr(config, key, value)
    return config


def _load_user_provided_config(
    user_config_path: Path,
    default_config_path: Path,
    config_class: type[BaseBenchmarkConfig],
) -> BaseBenchmarkConfig:
    """Load user-provided config file with inheritance from default config file.

    This function ensures that a user-provided config file inherits from a default
    config file, allowing users to override only the parameters they need
    while keeping all the default values from the base configuration.

    Args:
        user_config_path: Path to the user-provided configuration file
        default_config_path: Path to the default configuration file
        config_class: The benchmark config class to instantiate (e.g., ServingBenchmarkConfig)

    Returns:
        Config instance with inherited and overridden values
    """
    # Load the user config file
    with open(user_config_path, encoding="utf-8") as f:
        user_config_dict = yaml.safe_load(f)

    if not isinstance(user_config_dict, dict):
        raise ValueError(
            f"User configuration file {user_config_path} must contain a dictionary at the top level"
        )
    elif config_class._config_file_section_name not in user_config_dict:
        logger.warning(
            f"Cannot find {config_class._config_file_section_name} section in user configuration file {user_config_path}"
            f"Will not override benchmark config values from default config"
        )

    # Load the default config file
    with open(default_config_path, encoding="utf-8") as f:
        default_config_dict = yaml.safe_load(f)

    if not isinstance(default_config_dict, dict):
        raise ValueError(
            f"Default configuration file {default_config_path} must contain a dictionary at the top level"
        )

    # Merge the configs: user config overrides default config
    merged_config_dict = deep_merge_max_configs(
        default_config_dict, user_config_dict
    )

    # Resolve any depends_on paths relative to the default config file location
    # This is necessary because user provided configs may not have context on where
    # the "base" configs are located. This reference is only held in the default config file.
    if "depends_on" in merged_config_dict:
        depends_on_path = Path(merged_config_dict["depends_on"])
        if not depends_on_path.is_absolute():
            # Resolve relative to the default config file location
            merged_config_dict["depends_on"] = str(
                default_config_path.parent / depends_on_path
            )

    # Create a temporary config file with the merged content
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as temp_file:
        yaml.dump(merged_config_dict, temp_file)
        temp_config_path = temp_file.name

    try:
        # Load the merged config using the standard MAXConfig mechanism
        config = config_class.from_config_file(temp_config_path)
        return config
    finally:
        # Clean up the temporary file
        Path(temp_config_path).unlink(missing_ok=True)


def parse_benchmark_args(
    config_class: type[BaseBenchmarkConfig],
    default_config_path: Path,
    description: str,
    args: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """Parse command line arguments for benchmark entrypoints with config file inheritance.

    This function first parses a preliminary argument to get the config file path,
    then loads the appropriate configuration and re-parses with the loaded config as defaults.

    Its main purpose is to handle user provided config files which override params
    of a particular benchmark entrypoint.

    Args:
        config_class: The benchmark config class to instantiate (e.g., ServingBenchmarkConfig)
        default_config_path: Path to the default configuration file. For benchmark_serving.py,
        this should be the path to the serving_config.yaml file.
        description: Description for the argument parser
        args: Command line arguments to parse. If None, parse from sys.argv.

    Returns:
        Parsed arguments namespace with config file values as defaults
    """

    # Parse the config file argument first
    config_file_path, remaining_args = (
        _resolve_user_provided_config_file_cli_arg(args=args)
    )

    if config_file_path is None:
        logger.info(
            f"No configuration file path provided, using default {default_config_path} file"
        )
        benchmark_config = config_class.from_config_file(default_config_path)
    else:
        # Check if user provided the same file as default
        if config_file_path.resolve() == default_config_path.resolve():
            logger.info(f"Using default configuration file: {config_file_path}")
            benchmark_config = config_class.from_config_file(config_file_path)
        else:
            logger.info(
                f"Using user-provided configuration file: {config_file_path} (will inherit from {default_config_path})"
            )
            # Load the user config file and ensure it inherits from default config
            benchmark_config = _load_user_provided_config(
                config_file_path, default_config_path, config_class
            )

    # Create parser using the enhanced MAXConfig functionality
    # When a config file is loaded, only require parameters that are not provided in the config
    required_fields = config_class.get_default_required_fields()
    provided_required_fields = set()

    for field_name in required_fields:
        if hasattr(benchmark_config, field_name):
            field_value = getattr(benchmark_config, field_name)
            # Consider a field as "provided" if it has a non-None, non-empty value
            if field_value is not None and field_value != "":
                provided_required_fields.add(field_name)

    # Only require fields that are not provided in the config
    still_required_fields = required_fields - provided_required_fields

    parser = benchmark_config.cli_arg_parsers(
        description=description, required_params=still_required_fields
    )
    # This is added only for its help message. It's a no-op and not actually used for parsing
    # since it's done in the section above.
    parser = _add_config_file_arg_to_parser(parser)
    # Parse the remaining arguments with the loaded config as defaults
    return parser.parse_args(args=remaining_args)
