# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""ISTFT Head implementation for XCodec2 (stub implementation)."""

from __future__ import annotations

import max.nn as nn
from max.graph import TensorValue, ops

from .config import XCodec2Config


class ISTFTHead(nn.Module):
    """ISTFT Head module for XCodec2.

    This is a stub implementation that will be replaced with the full
    ISTFT implementation (complex spectrogram processing + inverse STFT) in a
    future PR.
    """

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.config = config
        self.dim = config.hidden_dim  # 1024
        self.n_fft = config.n_fft  # 1280
        self.hop_length = config.hop_length  # 320

        # self.n_fft are for the frequency bins of the STFT, and the +2 comes
        # from the magnitude and phase.
        out_dim = self.n_fft + 2  # 1282
        self.out = nn.Linear(
            self.dim, out_dim, dtype=config.dtype, device=config.device
        )

        # TODO: Implement full ISTFT in future PR:
        # - ISTFT module with Hann window
        # - Complex spectrogram processing
        # - Overlap-add reconstruction

    def __call__(self, x: TensorValue) -> tuple[TensorValue, TensorValue]:
        """Forward pass of ISTFTHead (stub implementation).

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Tuple of (audio, stft_pred)
            - audio: Audio waveform of shape (batch, 1, audio_length)
            - stft_pred: STFT predictions of shape (batch, out_dim, seq_len)
        """
        # Predict STFT coefficients
        x_pred = self.out(x)  # (batch, seq_len, out_dim)
        x_pred = x_pred.transpose(1, 2)  # (batch, out_dim, seq_len)

        # TODO: Replace with real ISTFT processing
        # For now, generate dummy audio of appropriate length
        batch_size, _, seq_len = x_pred.shape

        audio_length = (seq_len - 1) * self.hop_length + self.n_fft

        dummy_audio = ops.constant(
            0.0, dtype=self.config.dtype, device=self.config.device
        ).broadcast_to((batch_size, 1, audio_length))

        return dummy_audio, x_pred
