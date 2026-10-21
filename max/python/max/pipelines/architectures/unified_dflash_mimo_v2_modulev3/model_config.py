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
"""KV trees and drafter settings for the ModuleV3 MiMo-V2.6-Flash + DFlash
pipeline.

The drafter's context cache is always the last KV group, after the target's
``sliding_attention`` and ``full_attention``, which the fused speculative
graph nests under ``target`` -- the same tree the graph-API build takes, so
Mach's block-spec adapter binds either export unchanged.
"""

from __future__ import annotations

from dataclasses import replace

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParams,
    KVConnectorType,
    MultiKVCacheParams,
)
from max.pipelines.lib import (
    KVCacheConfig,
    PipelineConfig,
)
from transformers import AutoConfig
from typing_extensions import override

from ..dflash_mimo_v2.drafter import (
    drafter_config,
    drafter_kv_params,
    mimo_dflash_draft_width,
    read_dflash_config,
    repo_file,
    sampleable_vocab_size,
)
from ..mimo_v2_modulev3.model_config import MiMoV2Config

__all__ = [
    "DRAFT",
    "UnifiedDflashMiMoV2Config",
    "drafter_config",
    "drafter_kv_params",
    "mimo_dflash_draft_width",
    "read_dflash_config",
    "repo_file",
    "sampleable_vocab_size",
]


DRAFT = "draft"
"""The drafter's KV group, the tail of every MiMo tree that has one."""


def with_num_draft_tokens(
    params: MultiKVCacheParams, num_draft_tokens: int
) -> MultiKVCacheParams:
    """``params`` with ``num_draft_tokens`` on every leaf.

    The graph-API unified package owned this; the generic composition on
    main stamps the count itself, and this package builds its own fused
    tree, so it lives here now.
    """
    children = {}
    for name, leaf in params.children.items():
        assert isinstance(leaf, KVCacheParams)
        children[name] = replace(leaf, num_draft_tokens=num_draft_tokens)
    return MultiKVCacheParams.from_params(children)


class UnifiedDflashMiMoV2Config(MiMoV2Config):
    """The target's config, with the drafter's group in its KV tree.

    ``kv_params`` is ``{"target": {sliding, full}, "draft": ...}``, so memory
    planning prices the drafter's cache.
    """

    @override
    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        *,
        allow_kv_head_replication: bool = False,
    ) -> MultiKVCacheParams:
        """Builds ``{"target": {sliding, full}, "draft": ...}``.

        Raises:
            ValueError: If the cache is configured for dKV.
        """
        # The drafter's 1,024-key window gives the tree a second sliding
        # width beside the target's, which DKVConnector refuses: its touch
        # windows every sliding group by one width. The window saves real
        # memory, since the drafter's context is 20 KB a token, so DFlash on
        # MAX runs MiMo without dKV.
        if kv_cache_config.kv_connector_config.type == KVConnectorType.dkv:
            raise ValueError(
                "MiMo-V2 DFlash cannot use the dKV connector: the drafter's"
                " context window and the target's sliding window are two"
                " sliding widths, and dKV serves one."
            )
        draft_model = pipeline_config.draft_model
        assert draft_model is not None
        dflash = read_dflash_config(
            repo_file(draft_model.huggingface_weight_repo, "config.json")
        )
        draft = drafter_kv_params(
            dflash, pipeline_config, devices, kv_cache_config
        )
        target = with_num_draft_tokens(
            super().construct_kv_params(
                huggingface_config,
                pipeline_config,
                devices,
                kv_cache_config,
                cache_dtype,
                allow_kv_head_replication=allow_kv_head_replication,
            ),
            draft.num_draft_tokens,
        )
        return MultiKVCacheParams.from_params({"target": target, DRAFT: draft})
