# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Main XCodec2 model classes."""

from __future__ import annotations

import max.nn as nn
from max.graph import TensorValue, ops

from .config import XCodec2Config
from .istft import ISTFTHead
from .residual_fsq import ResidualFSQ
from .vocosbackbone import VocosBackbone


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


class XCodec2Model(nn.Module):
    """XCodec2 model for Llasa TTS.

    Implements only the decode_code functionality needed by Llama-based TTS.
    Converts speech tokens from Llama into audio waveforms.
    """

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.config = config
        self.generator = CodecDecoderVocos(config)
        self.fc_post_a = nn.Linear(
            config.vq_dim,  # 2048
            config.hidden_dim,  # 1024
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
            ops.squeeze(vq_code.transpose(1, 2), 1)  # (batch, seq_len)
        )  # Returns (batch, seq_len, vq_dim)

        # Step 2: Apply linear transformation (fc_post_a)
        vq_post_emb = self.fc_post_a(
            vq_post_emb
        )  # (batch, seq_len, hidden_dim)

        # Step 3: Generate audio through backbone + ISTFT head
        audio, _ = self.generator(vq_post_emb)  # (batch, 1, audio_length)

        # Return audio with batch dimension only: (batch, audio_length)
        return ops.squeeze(audio, 1)  # Remove channel dimension
