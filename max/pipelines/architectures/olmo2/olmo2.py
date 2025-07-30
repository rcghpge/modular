# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Build an Olmo2 model that uses continuous or paged kv-caching"""

import functools
from typing import Callable, Union

from max.dtype import DType
from max.graph.quantization import QuantizationEncoding
from max.nn import (
    MLP,
    Embedding,
    Linear,
    Llama3RotaryEmbedding,
    Module,
    RMSNorm,
    Transformer,
)
from max.nn.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheStrategy,
)
from max.pipelines.architectures.llama3.llama3 import StackedMLP
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.architectures.olmo2.layers.attention import Olmo2Attention

from .layers.transformer import Olmo2TransformerBlock


class Olmo2(Transformer):
    def __init__(self, config: Llama3Config) -> None:
        assert len(config.devices) == 1
        rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            head_dim=config.kv_params.head_dim,
            interleaved=config.interleaved_rope_weights,
            scaling_params=config.rope_scaling_params,
            device=config.devices[0],
        )

        # Select norm layer class.
        create_norm: Callable[..., Module]
        if config.norm_method == "rms_norm":
            if config.rms_norm_eps is None:
                raise ValueError(
                    "rms_norm_eps cannot be None for model that uses RMSNorm."
                )

            if config.norm_dtype is None:
                norm_dtype = config.dtype
            else:
                norm_dtype = config.norm_dtype

            create_norm = functools.partial(
                RMSNorm,
                config.hidden_size,
                dtype=norm_dtype,
                eps=config.rms_norm_eps,
            )
        else:
            raise ValueError(
                "norm_method cannot be other than RMSNorm for Olmo2."
            )

        # Select linear layer class.
        linear_cls: Callable[..., Linear]
        linear_cls = functools.partial(
            Linear, float8_config=config.float8_config
        )
        if config.stacked_mlp and config.float8_config:
            msg = "StackedMLP and float8 are not compatible"
            raise ValueError(msg)
        mlp_cls = (
            StackedMLP
            if config.stacked_mlp
            else functools.partial(MLP, float8_config=config.float8_config)
        )
        attention_cls: Callable[..., Olmo2Attention]
        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            assert config.quantization_config is not None
            assert not config.attention_bias, (
                "Attention bias is not supported for GPTQAttentionWithRope."
            )
            raise NotImplementedError(
                "GPTQ Olmo2Attention is not implemented yet"
            )
        elif config.model_quantization_encoding is not None:
            assert not config.attention_bias, (
                "Attention bias is not supported for GGUFQAttentionWithRope."
            )
            raise NotImplementedError(
                "GGUFQ Olmo2Attention is not implemented yet"
            )
        else:
            attention_cls = functools.partial(
                Olmo2Attention,
                scale=config.attention_multiplier,
                has_bias=config.attention_bias,
            )

        layers = [
            Olmo2TransformerBlock(
                attention=attention_cls(
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    layer_idx=i,
                    dtype=config.dtype,
                    rope=rope,
                    linear_cls=linear_cls,
                    devices=config.devices,
                ),
                mlp=mlp_cls(
                    config.dtype,
                    config.model_quantization_encoding,
                    config.hidden_size,
                    config.intermediate_size,
                    config.devices,
                    linear_cls,
                ),
                post_attention_layer_norm=create_norm(),
                post_feedforward_layer_norm=create_norm(),
                residual_multiplier=config.residual_multiplier,
            )
            for i in range(config.num_hidden_layers)
        ]

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype
        embedding_output_quantization = config.model_quantization_encoding
        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            embedding_output_dtype = DType.bfloat16
            embedding_output_quantization = None
        if config.float8_config and config.float8_config.embedding_output_dtype:
            embedding_output_dtype = config.float8_config.embedding_output_dtype
        embedding_layer = Embedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices[0],
            quantization_encoding=embedding_output_quantization,
        )
        output = Linear(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            config.devices[0],
            quantization_encoding=embedding_output_quantization,
        )

        if config.tie_word_embeddings:
            output.set_shared_weight("weight", embedding_layer.weight)

        kv_collection_cls: Union[
            type[FetchContinuousBatchingKVCacheCollection],
            type[FetchPagedKVCacheCollection],
        ]
        if config.kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS:
            kv_collection_cls = FetchContinuousBatchingKVCacheCollection
        elif config.kv_params.cache_strategy == KVCacheStrategy.PAGED:
            kv_collection_cls = FetchPagedKVCacheCollection
        else:
            raise ValueError(
                "Unsupported caching strategy "
                + str(config.kv_params.cache_strategy)
            )

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=create_norm(),
            output=output,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            kv_collection_constructor=kv_collection_cls(
                config.kv_params, num_layers=config.num_hidden_layers
            ),
            return_logits=config.return_logits,
            embedding_multiplier=config.embedding_multiplier,
            logits_postprocessor=config.logits_postprocessor,
        )
