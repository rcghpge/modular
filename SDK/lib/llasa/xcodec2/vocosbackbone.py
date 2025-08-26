# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""VocosBackbone implementation for XCodec2."""

from __future__ import annotations

import max.nn as nn
from max.graph import TensorValue

from .config import XCodec2Config
from .layers import ResnetBlock
from .transformer import TransformerBlock


class VocosBackbone(nn.Module):
    """Vocos backbone module for XCodec2.

    Implements the full VocosBackbone from the original implementation:
    - Conv1d embedding layer
    - ResNet blocks (prior_net)
    - Transformer blocks with rotary embeddings
    - Final layer normalization
    - ResNet blocks (post_net)

    With complex transpose operations to handle Conv1d vs Transformer format differences.
    """

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim  # 1024
        self.depth = config.depth  # 12
        self.heads = config.heads  # 16
        self.pos_emb_dim = config.pos_emb_dim  # 64

        # 1. Embedding layer (Conv1D)
        self.embed = nn.Conv1D(
            kernel_size=7,
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            padding=3,
            dtype=config.dtype,
            device=config.device,
        )

        # 2. Prior ResNet blocks (before transformers)
        prior_blocks = []
        for _ in range(2):  # 2 ResNet blocks
            prior_blocks.append(
                ResnetBlock(
                    in_channels=self.hidden_dim,
                    dtype=config.dtype,
                    device=config.device,
                )
            )
        self.prior_net = nn.LayerList(prior_blocks)

        # 3. Rotary positional embeddings
        self.time_rotary_embed = nn.Llama3RotaryEmbedding(
            dim=self.pos_emb_dim,  # 64
            n_heads=self.heads,  # 16
            theta=10000.0,  # Standard RoPE theta
            max_seq_len=2048,  # Reasonable max sequence length
            device=config.device,
            head_dim=self.pos_emb_dim,  # 64
            interleaved=False,
        )

        # 4. Transformer blocks
        transformer_blocks = []
        for _ in range(self.depth):  # 12 blocks
            transformer_blocks.append(
                TransformerBlock(
                    dim=self.hidden_dim,
                    n_heads=self.heads,
                    rotary_embed=self.time_rotary_embed,
                    dtype=config.dtype,
                    device=config.device,
                )
            )
        self.transformers = nn.LayerList(transformer_blocks)

        # 5. Final layer normalization
        self.final_layer_norm = nn.LayerNorm(
            dims=self.hidden_dim,
            device=config.device,
            dtype=config.dtype,
            eps=1e-6,
        )

        # 6. Post ResNet blocks (after transformers)
        post_blocks = []
        for _ in range(2):  # 2 ResNet blocks
            post_blocks.append(
                ResnetBlock(
                    in_channels=self.hidden_dim,
                    dtype=config.dtype,
                    device=config.device,
                )
            )
        self.post_net = nn.LayerList(post_blocks)

    def __call__(self, x: TensorValue) -> TensorValue:
        """Forward pass of VocosBackbone.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        # All operations expect (batch, seq_len, hidden_dim) format
        x = self.embed(x)
        x = self.prior_net(x)
        x = self.transformers(x)
        x = self.post_net(x)
        x = self.final_layer_norm(x)
        return x
