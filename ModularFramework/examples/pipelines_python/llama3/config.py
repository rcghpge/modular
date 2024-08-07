# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""All configurable parameters for Llama3."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class Llama3Config:
    weight_path: Optional[Union[str, Path]] = None
    version: float = 3.1
    quantization_encoding: str = "q4_k"

    max_length: int = 512
    max_new_tokens: int = -1

    batch_size: int = 1
    # TODO: Read from GGUF file.
    seq_len: int = 4096
    n_heads: int = 24
    n_layers: int = 32
    n_kv_heads: int = 8
    head_dim: int = 8

    def remote_weight_location(self):
        if self.weight_path is not None:
            return self.weight_path
        if self.version == 3:
            if self.quantization_encoding == "q4_0":
                return "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
            elif self.quantization_encoding == "q4_k":
                return "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
            elif self.quantization_encoding == "q6_k":
                return "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q6_K.gguf"
            elif self.quantization_encoding == "bfloat16":
                return "https://huggingface.co/ddh0/Meta-Llama-3-8B-Instruct-bf16-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-bf16.gguf"
            elif self.quantization_encoding == "float32":
                return "https://huggingface.co/brendanduke/Llama-3-8B-f32.gguf/resolve/main/llama3-8b-f32.gguf"
            else:
                raise ValueError(
                    f"quantization of {self.quantization_encoding} not"
                    f" supported for version {self.version}"
                )
        elif self.version == 3.1:
            if self.quantization_encoding == "q4_0":
                return "https://huggingface.co/kaetemi/Meta-Llama-3.1-8B-Q4_0-GGUF/resolve/main/meta-llama-3.1-8b-q4_0.gguf"
            elif self.quantization_encoding == "q4_k":
                return "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
            elif self.quantization_encoding == "q6_k":
                return "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"
            elif self.quantization_encoding == "bfloat16":
                return "https://huggingface.co/bullerwins/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-bf16.gguf"
            elif self.quantization_encoding == "float32":
                return "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-f32.gguf"
            else:
                raise ValueError(
                    f"quantization of {self.quantization_encoding} not"
                    f" supported for version {self.version}"
                )
        else:
            raise ValueError(f"unsupported model version {self.version}")
