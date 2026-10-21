# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""The Eagle3 heads over a DeepseekV3: their configs and module builders.

The heads also run over Kimi K2.5, whose language model is a DeepseekV3.
A head runs only beside its target, inside a unified spec-decode graph.
Each head's config builds its module for that graph from the head's own
checkpoint and its target's config: an MLA head in a one-layer view of the
target's cache, or a Llama-style MHA head in a cache of its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, replace
from typing import TypeVar

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData, load_weights
from max.nn.kv_cache import KVCacheParams
from max.nn.transformer import ReturnHiddenStates
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithBoundedMaxSeqLen,
)
from max.pipelines.lib.interfaces.pipeline_model import (
    ModelLoadContext,
)
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.pipelines.speculative import DraftAliases, validate_draft_state_dict
from transformers import AutoConfig
from typing_extensions import Self

from ..deepseekV3.model import DeepseekV3Model
from ..deepseekV3.model_config import DeepseekV3Config
from ..deepseekV3.weight_adapters import convert_safetensor_state_dict
from ..eagle_common.eagle_mha_draft import Eagle3MHADraft, Eagle3MHADraftConfig
from ..kimik2_5.unified_eagle_mha_pipeline_model import (
    _build_mha_draft_config,
    _infer_fc_input_multiplier,
)
from .eagle3_draft import Eagle3DeepseekV3

__all__ = [
    "Eagle3DeepseekV3DraftArchConfig",
    "Eagle3DeepseekV3DraftConfig",
    "Eagle3MHADeepseekV3DraftArchConfig",
    "extract_eagle_aux_layer_ids",
]

logger = logging.getLogger("max.pipelines")

_HeadT = TypeVar("_HeadT", Eagle3DeepseekV3, Eagle3MHADraft)


def extract_eagle_aux_layer_ids(
    hf_config: AutoConfig,
) -> list[int] | None:
    """Extract ``eagle_aux_hidden_state_layer_ids`` from a HuggingFace config.

    The IDs live inside an ``eagle_config`` sub-dict/object that is present on
    the *draft* checkpoint's config.
    """
    eagle_config = getattr(hf_config, "eagle_config", None)
    if eagle_config is None:
        return None
    raw = (
        eagle_config.get("eagle_aux_hidden_state_layer_ids", [])
        if isinstance(eagle_config, dict)
        else getattr(eagle_config, "eagle_aux_hidden_state_layer_ids", [])
    )
    raw_list = list(raw)
    if not raw_list:
        return None
    if any(i <= 0 for i in raw_list):
        raise ValueError(
            "eagle_aux_hidden_state_layer_ids must contain positive ids "
            "(capturing layer-0's input = raw token embeddings is not yet "
            f"wired in MAX). Got {raw_list}."
        )
    return [i - 1 for i in raw_list]


@dataclass
class Eagle3DeepseekV3DraftConfig(DeepseekV3Config):
    """The target's config, as the head runs it."""

    target_layer_ids: list[int] = field(default_factory=list)
    """The target layers the head fuses, as MAX indices."""


@dataclass
class Eagle3DeepseekV3DraftArchConfig(ArchConfigWithBoundedMaxSeqLen):
    """The draft architecture's config: its sequence bound and KV leaf."""

    max_seq_len: int

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Records the sequence length the pipeline resolved."""
        del pipeline_config, model_config
        return cls(max_seq_len=max_seq_len)

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Returns a one-layer view of the target's cache, for the one
        decoder layer the head has."""
        del huggingface_config
        target = DeepseekV3Config.construct_kv_params(
            pipeline_config.model.huggingface_config.get_text_config(),
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )
        assert isinstance(target, KVCacheParams)
        return replace(target, num_layers=1)

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[Eagle3DeepseekV3, Eagle3DeepseekV3DraftConfig]:
        """Builds the head from the draft checkpoint ``context`` names.

        It embeds through its target's embedding, so spec decoding shares
        that onto it, and its target's head too when the checkpoint ships
        none.
        """
        state_dict = _draft_state_dict(context)
        config = _draft_config(context, state_dict)
        if context.kv_params is not None:
            assert isinstance(context.kv_params, KVCacheParams)
            config.kv_params = context.kv_params
        return _loaded(Eagle3DeepseekV3(config), state_dict), config


@dataclass
class Eagle3MHADeepseekV3DraftArchConfig(ArchConfigWithBoundedMaxSeqLen):
    """The MHA head's config: its sequence bound and its own KV leaf."""

    max_seq_len: int

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Records the sequence length the pipeline resolved."""
        del pipeline_config, model_config
        return cls(max_seq_len=max_seq_len)

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Returns the head's one-layer cache, in bfloat16.

        MHA flash attention requires q/k/v and KV cache dtypes to match,
        and the head computes in bfloat16 whatever the target caches in.
        """
        if cache_dtype != DType.bfloat16:
            logger.warning(
                "Draft KV cache dtype forced to bfloat16 (target uses "
                f"{cache_dtype}). MHA flash attention requires q/k/v "
                "and KV cache dtypes to match; the draft module emits bf16."
            )
        speculative = pipeline_config.speculative
        return kv_cache_config.to_params(
            dtype=DType.bfloat16,
            n_kv_heads=int(huggingface_config.num_key_value_heads),
            head_dim=int(huggingface_config.head_dim),
            num_layers=1,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            is_mla=False,
            num_q_heads=int(huggingface_config.num_attention_heads),
            speculative_method=speculative.speculative_method
            if speculative
            else None,
            num_draft_tokens=(speculative.num_speculative_tokens or 0)
            if speculative
            else 0,
        )

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[Eagle3MHADraft, Eagle3MHADraftConfig]:
        """Builds the MHA head from the draft checkpoint ``context`` names.

        A checkpoint that doesn't name the target layers it was trained on
        gets layers spread across the target, which ``fc`` fuses as many of
        as its shape says.

        Raises:
            ValueError: If the named layers don't match ``fc``'s width.
        """
        draft_model = context.pipeline_config.draft_model
        assert draft_model is not None
        draft_hf = draft_model.huggingface_config
        state_dict = _draft_state_dict(context)
        target = _target_config(context)
        fc_input_multiplier = _infer_fc_input_multiplier(
            state_dict, hidden_size=int(draft_hf.hidden_size)
        )
        layer_ids = extract_eagle_aux_layer_ids(draft_hf)
        if layer_ids is None:
            target_layers = int(target.num_hidden_layers)
            layer_ids = _default_aux_layer_ids(
                target_layers, fc_input_multiplier
            )
            logger.warning(
                "Draft HF config has no "
                "'eagle_config.eagle_aux_hidden_state_layer_ids'. "
                "Falling back to evenly-spaced defaults for "
                f"{fc_input_multiplier}-way fusion over {target_layers} "
                f"target layers: {layer_ids}. Override by setting "
                "``eagle_config.eagle_aux_hidden_state_layer_ids`` on "
                "the draft HF config if the training-time IDs are "
                "known."
            )
        if len(layer_ids) != fc_input_multiplier:
            raise ValueError(
                f"Draft fc fuses {fc_input_multiplier} target hidden states "
                f"but the target is configured to capture {len(layer_ids)} "
                f"(eagle_aux_hidden_state_layer_ids={layer_ids})."
            )
        assert isinstance(context.kv_params, KVCacheParams)
        config = _build_mha_draft_config(
            draft_hf,
            target_config=target,
            devices=target.devices,
            data_parallel_degree=target.data_parallel_degree,
            fc_input_multiplier=fc_input_multiplier,
            kv_params=context.kv_params,
            sliding_window_override=draft_model.sliding_window,
        )
        config.target_layer_ids = layer_ids
        return _loaded(Eagle3MHADraft(config), state_dict), config


def _draft_state_dict(context: ModelLoadContext) -> dict[str, WeightData]:
    pipeline_config = context.pipeline_config
    assert pipeline_config.draft_model is not None
    return parse_state_dict_from_weights(
        pipeline_config,
        context.weights,
        context.adapter,
        hf_config=pipeline_config.draft_model.huggingface_config,
    )


def _loaded(draft: _HeadT, state_dict: dict[str, WeightData]) -> _HeadT:
    """Loads a head, which inherits its target's ``lm_head`` if it ships
    none."""
    draft.has_own_lm_head = "lm_head.weight" in state_dict
    draft.load_state_dict(state_dict, weight_alignment=1, strict=False)
    validate_draft_state_dict(
        draft.raw_state_dict().keys(),
        state_dict.keys(),
        DraftAliases(always=("embed_tokens.",), when_absent=("lm_head.",)),
    )
    return draft


def _target_config(context: ModelLoadContext) -> DeepseekV3Config:
    """The target's config, finalized from the target's own checkpoint.

    Raises:
        ValueError: If the head runs on more than one device without expert
            parallelism.
    """
    pipeline_config = context.pipeline_config
    n_devices = len(pipeline_config.model.device_specs)
    if n_devices > 1 and pipeline_config.runtime.ep_size != n_devices:
        raise ValueError("Only the EP strategy is supported.")
    return DeepseekV3Model.finalized_config(
        pipeline_config,
        parse_state_dict_from_weights(
            pipeline_config,
            load_weights(pipeline_config.model.resolved_weight_paths()),
            convert_safetensor_state_dict,
        ),
        max_seq_len=context.max_seq_len,
        max_batch_total_tokens=context.max_batch_total_tokens,
        return_logits=context.return_logits,
        return_hidden_states=ReturnHiddenStates.LAST,
    )


def _default_aux_layer_ids(
    num_target_layers: int, fc_input_multiplier: int
) -> list[int]:
    """Picks ``fc_input_multiplier`` aux capture layer IDs spread across
    a target with ``num_target_layers`` layers.

    Shifted one layer below vLLM's
    ``SupportsEagle3.get_eagle3_default_aux_hidden_state_layers`` default
    (``(1, num_layers // 2 - 1, num_layers - 4)`` for 3-way) — empirically
    a better match for the ``modularai/kimi-k2.5-eagle3`` checkpoint.
    """
    if num_target_layers <= 4:
        raise ValueError(
            f"Default aux layer IDs require >4 target layers, got "
            f"{num_target_layers}."
        )
    early = 1
    late = num_target_layers - 4
    if fc_input_multiplier == 2:
        return [early, late]
    if fc_input_multiplier == 3:
        return [early, num_target_layers // 2 - 1, late]
    raise ValueError(
        f"Unsupported fc_input_multiplier={fc_input_multiplier}; "
        "expected 2 or 3."
    )


def _draft_config(
    context: ModelLoadContext, draft_state_dict: dict[str, WeightData]
) -> Eagle3DeepseekV3DraftConfig:
    """The target's finalized config, with the head's RoPE and precision.

    Raises:
        ValueError: If the checkpoint names no target layers.
    """
    draft_model = context.pipeline_config.draft_model
    assert draft_model is not None
    draft_hf = draft_model.huggingface_config
    layer_ids = extract_eagle_aux_layer_ids(draft_hf)
    if layer_ids is None:
        raise ValueError(
            "eagle_aux_hidden_state_layer_ids must be present in the "
            "draft model's eagle_config for EAGLE3 hidden-state "
            "capture, but was not found in the draft HF config."
        )
    target = _target_config(context)
    config = Eagle3DeepseekV3DraftConfig(
        **{f.name: getattr(target, f.name) for f in fields(DeepseekV3Config)},
        target_layer_ids=layer_ids,
    )
    # The head may use different YarnRoPE parameters (e.g. beta_fast=1.0
    # vs the target's 32.0).
    draft_rope = getattr(draft_hf, "rope_scaling", None)
    if draft_rope is not None:
        config.rope_scaling = draft_rope
    # The head has a BF16 dense MLP: not quantized, not MoE.
    if (
        config.quant_config is not None
        and config.quant_config.is_fp4
        and not any("weight_scale" in key for key in draft_state_dict)
    ):
        logger.info(
            "Eagle3 draft weights are BF16 (no weight_scale found); "
            "disabling FP4 config for draft."
        )
        config.quant_config = None
        config.dtype = DType.bfloat16
        if config.ep_config is not None:
            config.ep_config.dispatch_dtype = DType.bfloat16
            config.ep_config.dispatch_quant_config = None
    return config
