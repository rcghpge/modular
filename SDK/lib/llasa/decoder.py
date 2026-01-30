# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""XCodec2 audio decoder for Llasa TTS pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from max.driver import Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType

from .xcodec2 import XCodec2
from .xcodec2.config import XCodec2Config
from .xcodec2.weight_loader import load_xcodec2_weights

logger = logging.getLogger(__name__)


def _xcodec2_decoder_graph_input_types(
    devices: list[Device],
) -> tuple[TensorType]:
    """Get the XCodec2 decoder graph input types.

    Args:
        devices: List of devices to use

    Returns:
        Tuple of input tensor types
    """
    device0 = devices[0]
    device_ref = DeviceRef(device0.label, device0.id)
    speech_ids_type = TensorType(
        DType.int32, ["batch_size", "seq_length"], device=device_ref
    )
    return (speech_ids_type,)


def _build_xcodec2_decoder_graph(
    devices: list[Device],
    xcodec2_decoder_weights: Path | str | None,
    xcodec2_config: XCodec2Config,
) -> tuple[Graph, dict[str, Any]]:
    """Build the XCodec2 decoder graph.

    Args:
        devices: List of devices to use
        xcodec2_decoder_weights: Path to XCodec2 weights file (HuggingFace model ID or local path)
        xcodec2_config: XCodec2 configuration

    Returns:
        Tuple of (graph, weights_dict)
    """
    # XCodec2 is non-causal, so we only use the first device
    devices = [devices[0]]
    graph_inputs = _xcodec2_decoder_graph_input_types(devices)

    nn_model = XCodec2(xcodec2_config)

    if xcodec2_decoder_weights:
        weights = load_xcodec2_weights(str(xcodec2_decoder_weights))
    else:
        raise ValueError(
            "xcodec2_decoder_weights must be provided. Random weights not supported for XCodec2."
        )

    nn_model.load_state_dict(
        weights,
        override_quantization_encoding=True,
        weight_alignment=1,
        strict=True,
    )

    with Graph("xcodec2_decoder", input_types=graph_inputs) as graph:
        vq_codes = graph.inputs[0].tensor
        audio_output = nn_model.decode_code(vq_codes)
        graph.output(audio_output)

    return graph, weights


def load_xcodec2_decoder(
    devices: list[Device],
    session: InferenceSession,
    xcodec2_decoder_weights: Path | str | None,
    xcodec2_config: XCodec2Config,
) -> Model:
    """Load the XCodec2 decoder model.

    Args:
        devices: List of devices to use
        session: Inference session
        xcodec2_decoder_weights: Path to XCodec2 weights (HuggingFace model ID or local path)
        xcodec2_config: XCodec2 configuration

    Returns:
        Compiled Model instance
    """
    logger.info("Building and compiling XCodec2 decoder...")
    before = time.perf_counter()

    weights_path: Path | str | None = None
    if xcodec2_decoder_weights:
        weights_path = (
            Path(xcodec2_decoder_weights)
            if isinstance(xcodec2_decoder_weights, str)
            and not xcodec2_decoder_weights.startswith("http")
            else None
        )
        if weights_path is None:
            weights_path = xcodec2_decoder_weights

    graph, weights = _build_xcodec2_decoder_graph(
        devices,
        weights_path,
        xcodec2_config,
    )

    model = session.load(graph, weights_registry=weights)
    after = time.perf_counter()
    logger.info(f"Building and compiling took {after - before:.6f} seconds")
    return model
