# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""ISTFT Head implementation for XCodec2."""

from __future__ import annotations

import max.nn as nn
from max.graph import TensorValue, ops

from .config import XCodec2Config


class ISTFT(nn.Module):
    """ISTFT implementation for XCodec2.

    Args:
        n_fft: Size of Fourier transform.
        hop_length: Distance between neighboring sliding window frames.
        win_length: Size of window frame and STFT filter.
        padding: Type of padding ("same" only - "center" not supported).
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        padding: str = "same",
    ) -> None:
        super().__init__()
        if padding != "same":
            raise ValueError("Only 'same' padding is supported.")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def __call__(self, spec: TensorValue) -> TensorValue:
        """Compute the Inverse Short Time Fourier Transform (ISTFT).

        Args:
            spec: Input complex spectrogram of shape (B, N, T, 2), where B is
                the batch size, N is the number of frequency bins, T is the
                number of time frames, and 2 is the interleaved real and
                imaginary parts.

        Returns:
            Reconstructed time-domain signal of shape (B, L), where L is the
            length of the output signal.
        """
        pad = (self.win_length - self.hop_length) // 2
        window = ops.hann_window(self.win_length, device=spec.device)

        assert spec.rank == 4, "Expected a 4D tensor as input"
        B = spec.shape[0]
        T = spec.shape[2]

        # Inverse FFT
        ifft = ops.irfft(
            spec,
            self.n_fft,
            axis=1,
            normalization="backward",
            input_is_complex=True,
        )
        ifft = ifft * window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = ops.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope (for proper ISTFT reconstruction)
        window_sq = (
            ops.reshape(window**2, (1, 1, -1))
            .broadcast_to((B, T, self.win_length))
            .transpose(1, 2)
        )

        window_envelope = ops.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).reshape((B, output_size))[:, pad:-pad]

        # Normalize by window envelope
        return y / window_envelope


class ISTFTHead(nn.Module):
    """ISTFT Head module for XCodec2.

    Predicts STFT complex coefficients and converts them to audio waveforms.
    """

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.config = config
        self.dim = config.hidden_dim  # 1024
        self.n_fft = config.n_fft  # 1280
        self.hop_length = config.hop_length  # 320

        # Linear layer to predict STFT coefficients
        # The +2 comes from the magnitude and phase predictions
        out_dim = self.n_fft + 2  # 1282
        self.out = nn.Linear(
            self.dim,
            out_dim,
            has_bias=True,
            dtype=config.dtype,
            device=config.device,
        )

        # ISTFT module for converting complex spectrogram to audio
        self.istft = ISTFT(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            padding="same",
        )

    def __call__(self, x: TensorValue) -> tuple[TensorValue, TensorValue]:
        """Forward pass of ISTFTHead.

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

        # Split into magnitude and phase predictions
        # Each: (batch, n_fft//2 + 1, seq_len) = (batch, 641, seq_len)
        mag, p = ops.chunk(x_pred, 2, axis=1)

        # Convert log-magnitude to magnitude
        mag = ops.exp(mag)
        # Safeguard to prevent excessively large magnitudes
        mag = ops.min(
            mag, ops.constant(1e2, dtype=mag.dtype, device=mag.device)
        )

        # Convert phase to real and imaginary components
        x_real = ops.cos(p)
        y_imag = ops.sin(p)

        # Create complex spectrogram in interleaved format for MAX
        # Shape: (batch, n_fft//2 + 1, seq_len, 2)
        x_real = ops.unsqueeze(x_real, -1)  # Add complex dimension
        y_imag = ops.unsqueeze(y_imag, -1)  # Add complex dimension
        complex_parts = ops.concat((x_real, y_imag), -1)  # Interleave real/imag

        # Apply magnitude to complex spectrogram
        S = ops.unsqueeze(mag, -1) * complex_parts  # (batch, 641, seq_len, 2)

        # Convert complex spectrogram to audio waveform
        audio = self.istft(S)  # (batch, audio_length)

        # Add channel dimension to match expected output format
        audio = ops.unsqueeze(audio, 1)  # (batch, 1, audio_length)

        return audio, x_pred
