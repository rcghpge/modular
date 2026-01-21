# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Main XCodec2 model classes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import Graph, TensorType, TensorValue, ops
from max.nn import legacy as nn

from .config import XCodec2Config
from .istft import ISTFTHead
from .residual_fsq import ResidualFSQ
from .vocosbackbone import VocosBackbone
from .weight_loader import load_xcodec2_weights


class CodecDecoderVocos(nn.Module):
    """CodecDecoderVocos implementation for Llasa TTS.

    Implements the decoder path of XCodec2: token embeddings → audio generation.
    Optimized for Llasa's use case where Llama provides speech tokens directly.
    """

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.config = config
        self.hop_length = config.hop_length
        self.quantizer = ResidualFSQ(config)
        self.backbone = VocosBackbone(config)
        self.head = ISTFTHead(config)

    def _embed_tokens(self, indices: TensorValue) -> TensorValue:
        """Convert token indices to embeddings using ResidualFSQ.

        Args:
            indices: Token indices of shape (batch, seq_len)

        Returns:
            Embeddings of shape (batch, seq_len, vq_dim)
        """
        return self.quantizer.get_output_from_indices(indices)

    def __call__(self, x: TensorValue) -> tuple[TensorValue, TensorValue]:
        """Forward pass of CodecDecoderVocos.

        Args:
            x: Input embeddings of shape (batch, seq_len, hidden_dim)

        Returns:
            Tuple of (audio, stft_pred)
            - audio: Audio waveform of shape (batch, 1, audio_length)
            - stft_pred: STFT predictions of shape (batch, out_dim, seq_len)
        """
        x = self.backbone(x)  # (batch, seq_len, hidden_dim)
        audio, stft_pred = self.head(
            x
        )  # (batch, 1, audio_length), (batch, out_dim, seq_len)
        return audio, stft_pred


class XCodec2(nn.Module):
    """XCodec2 core model architecture.

    Implements the decode_code functionality needed by Llama-based TTS.
    Converts speech tokens from Llama into audio waveforms.
    """

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.config = config
        self.generator = CodecDecoderVocos(config)
        self.fc_post_a = nn.Linear(
            config.vq_dim,  # 2048
            config.hidden_dim,  # 1024
            has_bias=True,
            dtype=config.dtype,
            device=config.device,
        )

    def __call__(self, *args, **kwargs):
        """This model is not meant to be called directly. Use decode_code() instead."""
        raise NotImplementedError(
            "Use decode_code() method instead of calling the model directly"
        )

    def decode_code(self, vq_code: TensorValue) -> TensorValue:
        """Decode speech token codes to audio waveform.

        This matches the signature from the original modeling_xcodec2.py decode_code method.

        Args:
            vq_code: Speech token indices of shape (batch, seq_len)

        Returns:
            Audio waveform of shape (batch, audio_length)
        """
        # Step 1: Get quantized embeddings from indices
        # vq_code: (batch, seq_len) -> transpose for quantizer -> (batch, 1, seq_len)
        if len(vq_code.shape) == 2:
            vq_code = ops.unsqueeze(vq_code, 1)  # (batch, 1, seq_len)

        vq_post_emb = self.generator._embed_tokens(
            ops.squeeze(vq_code.transpose(1, 2), 2)  # (batch, seq_len)
        )  # Returns (batch, seq_len, vq_dim)

        # Step 2: Apply linear transformation (fc_post_a)
        vq_post_emb = self.fc_post_a(
            vq_post_emb
        )  # (batch, seq_len, hidden_dim)

        # Step 3: Generate audio through backbone + ISTFT head
        audio, _ = self.generator(vq_post_emb)  # (batch, 1, audio_length)

        # Return audio with batch dimension only: (batch, audio_length)
        return ops.squeeze(audio, 1)  # Remove channel dimension


class XCodec2Model:
    """XCodec2 model wrapper for Llasa TTS.

    Provides speech token to audio conversion using XCodec2's decoder components.
    Handles model loading, graph compilation, and both TensorValue and NumPy interfaces.
    """

    def __init__(self, config: XCodec2Config, model_path: str | Path) -> None:
        """Initialize the XCodec2 model.

        Args:
            config: XCodec2 configuration
            model_path: HuggingFace model ID or path to model weights
        """
        self.config = config
        self.xcodec2 = XCodec2(config)

        weights = load_xcodec2_weights(str(model_path))
        self.xcodec2.load_state_dict(
            weights,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=True,
        )

        # Build and compile the decoding graph
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
            audio_output = self.xcodec2.decode_code(speech_tokens)
            graph.output(audio_output)

        self.compiled_model = self.session.load(
            graph, weights_registry=self.xcodec2.state_dict()
        )

    def decode_code(self, vq_code: TensorValue) -> TensorValue:
        """Decode speech token codes to audio waveform (TensorValue interface).

        Args:
            vq_code: Speech token indices of shape (batch, seq_len)

        Returns:
            Audio waveform of shape (batch, audio_length)
        """
        return self.xcodec2.decode_code(vq_code)

    def decode(self, speech_tokens: np.ndarray) -> np.ndarray:
        """Decode speech tokens to audio waveform (NumPy interface).

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

        input_tensor = Buffer.from_numpy(speech_tokens_batched.astype(np.int32))
        input_tensor = input_tensor.to(self.config.device.to_device())

        output_tensors = self.compiled_model.execute(input_tensor)

        audio_tensor = output_tensors[0]
        assert isinstance(audio_tensor, Buffer), (
            f"Expected Tensor, got {type(audio_tensor)}"
        )
        audio_numpy = audio_tensor.to_numpy()

        if not is_batched and audio_numpy.shape[0] == 1:
            audio_numpy = audio_numpy[0]

        return audio_numpy

    @property
    def sample_rate(self) -> int:
        """Audio sample rate (16kHz)."""
        return self.config.sample_rate

    @property
    def codec_tokens_per_sec(self) -> int:
        """Number of codec tokens per second (50)."""
        return self.config.codec_tokens_per_sec

    @property
    def duration_per_token(self) -> float:
        """Duration per token in seconds (1/50 = 0.02 seconds)."""
        return 1.0 / self.codec_tokens_per_sec
