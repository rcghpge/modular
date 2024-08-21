# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""All configurable parameters for Llama3."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union


class _SupportedEncodings(str, Enum):
    float32 = "float32"
    bfloat16 = "bfloat16"
    q4_0 = "q4_0"
    q4_k = "q4_k"
    q6_k = "q6_k"

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class _SupportedVersions(str, Enum):
    llama3 = "3"
    llama3_1 = "3.1"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


PRETRAINED_MODEL_WEIGHTS = {
    _SupportedVersions.llama3: {
        _SupportedEncodings.q4_0: "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf",
        _SupportedEncodings.q4_k: "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        _SupportedEncodings.q6_k: "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q6_K.gguf",
        _SupportedEncodings.bfloat16: "https://huggingface.co/ddh0/Meta-Llama-3-8B-Instruct-bf16-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-bf16.gguf",
        _SupportedEncodings.float32: "https://huggingface.co/brendanduke/Llama-3-8B-f32.gguf/resolve/main/llama3-8b-f32.gguf",
    },
    _SupportedVersions.llama3_1: {
        _SupportedEncodings.q4_0: "https://huggingface.co/kaetemi/Meta-Llama-3.1-8B-Q4_0-GGUF/resolve/main/meta-llama-3.1-8b-q4_0.gguf",
        _SupportedEncodings.q4_k: "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        _SupportedEncodings.q6_k: "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
        _SupportedEncodings.bfloat16: "https://huggingface.co/bullerwins/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-bf16.gguf",
        _SupportedEncodings.float32: "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-f32.gguf",
    },
}


@dataclass
class Llama3Config:
    weight_path: Optional[Union[str, Path]] = None
    """Path or URL of the model weights."""

    version: _SupportedVersions = _SupportedVersions.llama3_1
    """Llama version."""

    # TODO: Change default to "q4_k"
    quantization_encoding: _SupportedEncodings = _SupportedEncodings.float32
    """Quantization encoding type."""

    serialized_model_path: Optional[Union[str, Path]] = None
    """If specified, tries to load a serialized model from this path."""

    max_length: int = 512
    """Controls the maximum length of the text sequence (includes the input tokens)."""

    max_new_tokens: int = -1
    """Controls the maximum length of the text sequence (does not include the input tokens)."""

    batch_size: int = 1
    """Batch size of inputs to the model."""

    seq_len: int = 4096
    """Maximum length of the token sequence that can be processed by this model."""

    n_layers: int = 32
    """Number of MultiHeadAttention layers to use in this model."""

    n_heads: int = 32
    """Number of heads for the query to use in the MultiHeadAttention layers."""

    n_kv_heads: int = 8
    """Number of key and value heads to use in the MultiHeadAttention layers."""

    vocab_size: int = 128256
    """Number of tokens in the vocabulary."""

    hidden_dim: int = 4096
    """Hidden dimension of embedded tokens."""

    rope_theta: float = 500000.0
    """Rotary period hyperparameter for rope embeddings."""

    layer_norm_rms_epsilon: float = 1e-5
    """Epsilon value for layer norm calculation."""

    @staticmethod
    def help():
        return {
            "weight_path": (
                "Path or URL of the model weights. If not provided, default"
                " weights will be downloaded based on the version and"
                " quantization encoding."
            ),
            "version": "Llama version.",
            "quantization_encoding": (
                "The encoding to use for a datatype that can be quantized to a"
                " low bits per weight format."
            ),
            "max_length": (
                "Controls the maximum length of the text sequence (includes the"
                " input tokens)."
            ),
            "version": (
                "Controls the maximum length of the text sequence (does not"
                " include the input tokens)."
            ),
            "batch_size": "Batch size of inputs to the model.",
            "seq_len": (
                "Maximum length of the token sequence that can be processed by"
                " this model."
            ),
            "n_layers": (
                "Number of MultiHeadAttention layers to use in this model."
            ),
            "n_heads": (
                "Number of heads for the query to use in the MultiHeadAttention"
                " layers."
            ),
            "n_kv_heads": (
                "Number of key and value heads to use in the MultiHeadAttention"
                " layers."
            ),
            "vocab_size": "Number of tokens in the vocabulary.",
            "hidden_dim": "Hidden dimension of embedded tokens.",
            "rope_theta": "Rotary period hyperparameter for rope embeddings.",
            "layer_norm_rms_epsilon": (
                "Epsilon value for layer norm calculation."
            ),
        }

    def remote_weight_location(self):
        if self.weight_path is not None:
            return self.weight_path
        try:
            version_weights = PRETRAINED_MODEL_WEIGHTS[self.version]
        except KeyError:
            raise ValueError(f"unsupported model version {self.version}")
        try:
            return version_weights[self.quantization_encoding]
        except KeyError:
            raise ValueError(
                f"quantization of {self.quantization_encoding} not"
                f" supported for version {self.version}"
            )

    @property
    def head_dim(self):
        """Dimension of each head."""
        return self.hidden_dim // self.n_heads

    @property
    def kv_weight_dim(self):
        """Dimension of the key and value attention weights."""
        return self.head_dim * self.n_kv_heads
