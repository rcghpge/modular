# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Standardized response object for Pipeline Inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LogProbabilities:
    text_offset: list[int]
    """String offset of each decoded token."""

    token_log_probabilities: list[np.float32]
    """Probabilities of each token."""

    top_log_probabilities: list[dict[str, np.float32]]
    """Top tokens and their corresponding probabilities."""

    tokens: list[str]
    """Decoded tokens."""


@dataclass(frozen=True)
class TextResponse:
    """A base class for model response, specifically for Text model variants."""

    next_token: int | str
    """Encoded predicted next token."""

    log_probabilities: LogProbabilities | None = None
    """Log probabilities of each output token."""
