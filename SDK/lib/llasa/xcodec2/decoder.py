# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""High-level XCodec2 decoder wrapper for Llasa integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from max.driver import Tensor
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import Graph, TensorType, TensorValue

from .config import XCodec2Config
from .weight_loader import load_xcodec2_weights
from .xcodec2 import XCodec2Model


class XCodec2Decoder:
    """XCodec2 decoder wrapper for Llasa TTS.

    Provides speech token to audio conversion using XCodec2's decoder components.
    Optimized for Llama-generated speech tokens.
    """

    def __init__(self, config: XCodec2Config, model_path: str | Path) -> None:
        """Initialize the XCodec2 decoder.

        Args:
            config: XCodec2 configuration
            model_path: HuggingFace model ID or path to model weights
        """
        self.config = config
        self.xcodec2_model = XCodec2Model(config)

        weights = load_xcodec2_weights(str(model_path))
        self.xcodec2_model.load_state_dict(
            weights,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=True,
        )

        self._build_decode_graph()

    def _build_decode_graph(self) -> None:
        """Build and compile the decoding graph for numpy interface."""
        device = self.config.device
        self.session = InferenceSession(devices=[device.to_device()])

        speech_tokens_type = TensorType(
            DType.int32,
            ("batch_size", "seq_length"),
            device=self.config.device,
        )

        with Graph(
            "xcodec2_decoder",
            input_types=(speech_tokens_type,),
        ) as graph:
            speech_tokens = graph.inputs[0].tensor
            audio_output = self.xcodec2_model.decode_code(speech_tokens)
            graph.output(audio_output)

        self.compiled_model = self.session.load(
            graph, weights_registry=self.xcodec2_model.state_dict()
        )

    def decode_code(self, vq_code: TensorValue) -> TensorValue:
        """Decode speech token codes to audio waveform.

        Args:
            vq_code: Speech token indices of shape (batch, seq_len)

        Returns:
            Audio waveform of shape (batch, audio_length)
        """
        return self.xcodec2_model.decode_code(vq_code)

    @property
    def sample_rate(self) -> int:
        """Audio sample rate (16kHz)."""
        return self.config.sample_rate

    @property
    def codec_tokens_per_sec(self) -> int:
        """Number of codec tokens per second (50)."""
        return self.config.codec_tokens_per_sec

    def decode_numpy(self, speech_tokens: np.ndarray) -> np.ndarray:
        """Decode speech tokens to audio waveform (NumPy interface).

        Converts between NumPy arrays (used by AudioDecoder) and TensorValues (used by XCodec2Model).

        Args:
            speech_tokens: NumPy array of speech token indices, shape (num_tokens,) or (batch, num_tokens)

        Returns:
            Audio waveform as NumPy array, shape (num_samples,) or (batch, num_samples)
        """
        if speech_tokens.ndim == 1:
            speech_tokens_batched = speech_tokens[np.newaxis, :]
            is_batched = False
        else:
            speech_tokens_batched = speech_tokens
            is_batched = True

        input_tensor = Tensor.from_numpy(speech_tokens_batched.astype(np.int32))
        input_tensor = input_tensor.to(self.config.device.to_device())

        output_tensors = self.compiled_model.execute(input_tensor)

        audio_tensor = output_tensors[0]
        assert isinstance(audio_tensor, Tensor), (
            f"Expected Tensor, got {type(audio_tensor)}"
        )
        audio_numpy = audio_tensor.to_numpy()

        if not is_batched and audio_numpy.shape[0] == 1:
            audio_numpy = audio_numpy[0]

        return audio_numpy
