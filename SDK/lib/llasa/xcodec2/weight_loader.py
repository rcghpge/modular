# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Weight loading utilities for XCodec2."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from max.driver import Tensor
from max.graph.weights import SafetensorWeights

logger = logging.getLogger(__name__)


def _map_xcodec2_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Filter HuggingFace XCodec2 weights to only those needed by Llasa.

    Llasa only needs the decoder path since Llama handles text understanding.
    This filters out unused components and fixes PyTorch→MAX compatibility issues.

    Args:
        state_dict: HuggingFace model state dict

    Returns:
        Filtered state dict with only weights needed by Llasa
    """
    filtered_weights: dict[str, Any] = {}

    excluded_patterns = [
        "semantic_model.",
        "SemanticEncoder_module.",
        "CodecEnc.",
        "encoder.",
        "generator.backbone.embed.bias",
        "conv1.bias",
        "conv2.bias",
        "fc_prior",
        "fc_post_a.bias",
        "generator.head.istft.window",
    ]

    for weight_name, tensor_value in state_dict.items():
        if any(pattern in weight_name for pattern in excluded_patterns):
            continue

        # Handle shape transformation for PyTorch→MAX compatibility
        processed_tensor = tensor_value
        if "weight" in weight_name and len(tensor_value.shape) == 3:
            tensor_np = tensor_value.to_numpy()

            if "embed" in weight_name:
                transposed_np = np.transpose(tensor_np, (2, 0, 1))
                processed_tensor = Tensor.from_numpy(transposed_np)

            elif "conv" in weight_name:
                transposed_np = np.transpose(tensor_np, (2, 1, 0))
                processed_tensor = Tensor.from_numpy(transposed_np)

        filtered_weights[weight_name] = processed_tensor

    return filtered_weights


def load_xcodec2_weights(model_path: str) -> dict[str, Any]:
    """Load XCodec2 weights from HuggingFace model.

    Args:
        model_path: HuggingFace model ID (e.g. "HKUSTAudio/xcodec2")

    Returns:
        Mapped weights ready for loading into MAX XCodec2Model
    """
    try:
        weight_file = hf_hub_download(
            repo_id=model_path,
            filename="model.safetensors",
            cache_dir=None,
        )
        logger.info(f"Downloaded model.safetensors to: {weight_file}")
    except Exception as e:
        logger.error(
            f"Failed to download model.safetensors from {model_path}: {e}"
        )
        raise FileNotFoundError(
            f"Could not download model.safetensors from {model_path}"
        ) from e

    safetensor_weights = SafetensorWeights([Path(weight_file)])

    weights = {}
    for key, weight in safetensor_weights.items():
        weights[key] = weight.data().data

    mapped_weights = _map_xcodec2_state_dict(weights)

    return mapped_weights
