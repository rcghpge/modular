# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Standardized response object for Pipeline Inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LogProbabilities:
    token_log_probabilities: list[float]
    """Probabilities of each token."""

    top_log_probabilities: list[dict[int, float]]
    """Top tokens and their corresponding probabilities."""


@dataclass(frozen=True)
class TextResponse:
    """A base class for model response, specifically for Text model variants."""

    next_token: int | str
    """Encoded predicted next token."""

    log_probabilities: LogProbabilities | None = None
    """Log probabilities of each output token."""
