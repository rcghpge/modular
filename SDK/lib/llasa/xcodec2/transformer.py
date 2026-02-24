# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Transformer components for XCodec2."""

from __future__ import annotations

import max.nn as nn
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu


class MLP(nn.Module):
    """Multi-Layer Perceptron for XCodec2 transformer blocks."""

    def __init__(
        self,
        dim: int,
        dtype: DType = DType.float32,
        device: DeviceRef = DeviceRef.GPU(),
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            dim, 4 * dim, has_bias=False, dtype=dtype, device=device
        )
        self.fc2 = nn.Linear(
            4 * dim, dim, has_bias=False, dtype=dtype, device=device
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Forward pass of MLP.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        x = self.fc1(x)
        x = ops.silu(x)
        x = self.fc2(x)
        return x


class SimpleAttention(nn.Module):
    """Simple multi-head attention layer for XCodec2 transformer blocks.

    This implementation is a simplified variant of the attention used in
    large language models. It is designed for audio token processing,
    where full-sequence context is available and incremental decoding
    is unnecessary.

    Differences from LLM-style attention:
    - No KV caching: audio processing does not require step-by-step generation,
      so past keys/values are not stored.
    - No causal masking: tokens are free to attend to all positions in the
      sequence (non-autoregressive).
    - No layer indices: positional information is handled through rotary
      embeddings without requiring per-layer tracking.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        rotary_embed: nn.RotaryEmbedding,
        dtype: DType = DType.float32,
        device: DeviceRef = DeviceRef.GPU(),
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0, (
            f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        )

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rotary_embed = rotary_embed

        # Combined QKV projection
        self.c_attn = nn.Linear(
            dim, 3 * dim, has_bias=False, dtype=dtype, device=device
        )

        # Output projection
        self.c_proj = nn.Linear(
            dim, dim, has_bias=False, dtype=dtype, device=device
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Forward pass of attention.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape

        qkv = self.c_attn(x)  # (batch, seq_len, 3*dim)

        # Reshape for flash attention: (batch, seq_len, 3, n_heads, head_dim)
        qkv_reshaped = qkv.reshape(
            (batch_size, seq_len, 3, self.n_heads, self.head_dim)
        )

        # Split into Q, K, V with correct shape for flash_attention_gpu
        q = qkv_reshaped[:, :, 0, :, :]  # (batch, seq_len, n_heads, head_dim)
        k = qkv_reshaped[:, :, 1, :, :]  # (batch, seq_len, n_heads, head_dim)
        v = qkv_reshaped[:, :, 2, :, :]  # (batch, seq_len, n_heads, head_dim)

        # Apply rotary embeddings
        q = self.rotary_embed(q)  # Apply RoPE to queries
        k = self.rotary_embed(k)  # Apply RoPE to keys

        y = flash_attention_gpu(
            q,
            k,
            v,
            mask_variant=MHAMaskVariant.NULL_MASK,  # No masking for audio
            scale=1.0 / (self.head_dim**0.5),
        )  # (batch, seq_len, n_heads, head_dim)

        # Reshape back to (batch, seq_len, dim) - no transpose needed!
        y = y.reshape((batch_size, seq_len, self.dim))

        y = self.c_proj(y)
        return y


class TransformerBlock(nn.Module):
    """Transformer block for XCodec2."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        rotary_embed: nn.RotaryEmbedding,
        dtype: DType = DType.float32,
        device: DeviceRef = DeviceRef.GPU(),
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.att_norm = nn.RMSNorm(dim, dtype=dtype, eps=1e-6)
        self.ffn_norm = nn.RMSNorm(dim, dtype=dtype, eps=1e-6)
        self.att = SimpleAttention(
            dim, n_heads, rotary_embed, dtype=dtype, device=device
        )
        self.mlp = MLP(dim, dtype=dtype, device=device)

    def __call__(self, x: TensorValue) -> TensorValue:
        """Forward pass of TransformerBlock.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        x = x + self.att(self.att_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x
