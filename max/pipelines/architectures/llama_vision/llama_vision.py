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

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional, cast, final

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    DeviceRef,
    Dim,
    Graph,
    SymbolicDim,
    TensorType,
    TensorValue,
    ops,
)
from max.graph.weights import Weights, WeightsAdapter
from max.interfaces import InputContext
from max.interfaces.request import RequestID
from max.nn import LinearV1, ReturnLogits
from max.nn.kv_cache import (
    ContinuousBatchingKVCacheManager,
    KVCacheInputs,
    KVCacheInputSymbols,
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
    PaddedKVCacheInputs,
    PagedKVCacheManager,
    RaggedKVCacheInputs,
    build_max_lengths_tensor,
    estimate_kv_cache_size,
    infer_optimal_batch_size,
    load_kv_manager,
)
from max.nn.layer import Layer
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
    upper_bounded_default,
)
from max.support.math import ceildiv
from transformers import AutoConfig

from .language_model import CausalLanguageModel, instantiate_language_model
from .model_config import LlamaVisionConfig
from .vision_model import instantiate_vision_model

logger = logging.getLogger("max.pipelines")

# TODO(GEX-2071): Re-enable when parallel compilation works.
_DO_PARALLEL_COMPILATION = False


@dataclass
class MultimodalKVCacheInputSymbols(KVCacheInputSymbols):
    text_kv_input_symbols: KVCacheInputSymbols
    vision_kv_input_symbols: KVCacheInputSymbols


@dataclass
class MultimodalKVCacheInputs(KVCacheInputs):
    text_kv_cache_inputs: KVCacheInputs
    vision_kv_cache_inputs: KVCacheInputs


class MultimodalKVCacheManager(KVCacheManager):
    """A lightweight wrapper around text and vision KV managers.

    Note on runtime and graph build time return types:
    - Currently the multi modal KV manager doesn't support multiple devices.
      So all lists that should be of length num_devices will have length 1.
    - Individual modality KV cache managers return a 4-tuple of KV cache inputs.
      Since this is a pair of KV cache managers, it returns an 8-tuple,
      where the first 4 elements are the text KV cache inputs and the remaining
      4 elements are the vision KV cache inputs.
    - This 8-tuple applies for both input symbols and return KV cache inputs.
    - TODO(bduke): We should fix both multi-device and multi-modality using an
      extensible KVCacheInput type.
    """

    text_kv_manager: PagedKVCacheManager
    """KV cache manager for text inputs."""

    vision_kv_manager: PagedKVCacheManager
    """KV cache manager for image inputs."""

    def __init__(
        self,
        params: KVCacheParams,
        max_batch_size: Optional[int],
        text_max_seq_len: int,
        vision_max_seq_len: int,
        text_num_layers: int,
        vision_num_layers: int,
        devices: Sequence[Device],
        session: InferenceSession,
        available_cache_memory: int,
        page_size: int,
    ) -> None:
        assert max_batch_size, "Expected max_batch_size to be set"
        paged_text_kv_manager = load_kv_manager(
            params=params,
            max_batch_size=max_batch_size,
            max_seq_len=text_max_seq_len,
            num_layers=text_num_layers,
            devices=devices,
            available_cache_memory=available_cache_memory,
            page_size=page_size,
            session=session,
        )
        assert isinstance(paged_text_kv_manager, PagedKVCacheManager)
        self.text_kv_manager = paged_text_kv_manager

        # Assume the number of vision tokens is fixed per batch.
        # This is true until we support multi-image.

        # Round up to the nearest multiple of 128.
        # This is because the page size must be a multiple of tile size.
        page_size = ceildiv(vision_max_seq_len, 128) * 128

        self.vision_kv_params = KVCacheParams(
            dtype=params.dtype,
            n_kv_heads=params.n_kv_heads,
            head_dim=params.head_dim,
            enable_prefix_caching=params.enable_prefix_caching,
            enable_kvcache_swapping_to_host=params.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=params.host_kvcache_swap_space_gb,
            cache_strategy=KVCacheStrategy.PAGED,
            page_size=page_size,
            n_devices=params.n_devices,
        )

        # Compute the bytes for the vision KV cache.
        single_token_size_bytes = (
            2
            * vision_num_layers
            * params.n_kv_heads_per_device
            * params.head_dim
            * params.dtype.size_in_bytes
        )
        cache_memory_per_image = single_token_size_bytes * page_size
        cache_memory = cache_memory_per_image * max_batch_size

        # Always use paged KV cache for the vision KV projections.
        self.vision_kv_manager = PagedKVCacheManager(
            params=self.vision_kv_params,
            max_batch_size=max_batch_size,
            max_seq_len=vision_max_seq_len,
            num_layers=vision_num_layers,
            devices=devices,
            session=session,
            cache_memory=cache_memory,
            page_size=page_size,
        )

        # Call superclass after initializing modality KV managers since the
        # superclass ctor calls methods that use the modality KV managers.
        super().__init__(
            params,
            max_batch_size,
            text_max_seq_len,
            text_num_layers,
            devices,
            session,
            is_ragged=True,
        )

    @classmethod
    @final
    def estimated_memory_size(
        cls,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        available_cache_memory: int,
        devices: Sequence[Device],
        **kwargs: Any,
    ) -> int:
        """Returns the estimated total memory usage of the kv cache."""
        assert "num_vision_layers" in kwargs, "num_vision_layers must be set"
        num_vision_layers = kwargs["num_vision_layers"]
        assert "max_vision_seq_len" in kwargs, "max_vision_seq_len must be set"
        max_vision_seq_len = kwargs["max_vision_seq_len"]

        vision_kv_cache_size = (
            ContinuousBatchingKVCacheManager.estimated_memory_size(
                params,
                max_batch_size,
                max_vision_seq_len,
                num_vision_layers,
                available_cache_memory,
                devices,
            )
        )

        remaining_memory = available_cache_memory - vision_kv_cache_size
        if remaining_memory <= 0:
            return vision_kv_cache_size

        text_kv_cache_size = estimate_kv_cache_size(
            params,
            max_batch_size,
            max_seq_len,
            num_layers,
            remaining_memory,
            devices,
        )
        return vision_kv_cache_size + text_kv_cache_size

    @classmethod
    def infer_optimal_batch_size(
        cls,
        params: KVCacheParams,
        max_seq_len: int,
        num_layers: int,
        available_cache_memory: int,
        devices: Sequence[Device],
        **kwargs: Any,
    ) -> int:
        """Returns the estimated optimal batch size for the kv cache."""
        assert "num_vision_layers" in kwargs, "num_vision_layers must be set"
        num_vision_layers = kwargs["num_vision_layers"]
        assert "max_vision_seq_len" in kwargs, "max_vision_seq_len must be set"
        max_vision_seq_len = kwargs["max_vision_seq_len"]

        # figure out the relative sizes of caches based on KV Cache settings
        text_size_per_token = num_layers * max_seq_len
        vision_size_per_token = num_vision_layers * max_vision_seq_len
        text_to_vision_ratio = text_size_per_token / (
            text_size_per_token + vision_size_per_token
        )

        # divvy up our allocation based on this ratio
        text_cache_size = available_cache_memory * text_to_vision_ratio
        vision_cache_size = available_cache_memory - text_cache_size

        # infer the optimal batch size for each modality based on its cache size
        text_batch_size = infer_optimal_batch_size(
            params, max_seq_len, num_layers, text_cache_size, devices
        )
        vision_batch_size = (
            ContinuousBatchingKVCacheManager.infer_optimal_batch_size(
                params,
                max_vision_seq_len,
                num_vision_layers,
                vision_cache_size,
                devices,
            )
        )

        return min(text_batch_size, vision_batch_size)

    @final
    def fetch(
        self, batch: list[InputContext], num_steps: int = 1
    ) -> list[KVCacheInputs]:
        """Returns KV cache inputs for both modalities' KV managers."""
        # Here we call into the text KV manager's fetch method to update
        # its fetch metadata.
        text_fetch_results = self.text_kv_manager.fetch(batch, num_steps)[0]

        # For the vision KV manager, fetch metadata isn't applicable since
        # autoregressive generation is text only.
        active_batch_size = len(batch)
        cache_lengths_np = np.zeros(active_batch_size, np.uint32)

        max_seq_length = 0
        max_cache_length = 0

        device = self.vision_kv_manager.devices[0]
        for i, ctx in enumerate(batch):
            # Assumption: If start_idx is greater than 0, then it has encoded its
            # vision input and has the max image sequence length.
            # TODO(bduke): pass the vision sequence lengths in from next_token.

            # Omit validity checks on seq ids, which are done in the text fetch.
            cache_len = (
                self.vision_kv_manager.max_seq_len if ctx.start_idx > 0 else 0
            )
            if cache_len == 0:
                max_seq_length = self.vision_kv_manager.max_seq_len

            cache_lengths_np[i] = cache_len

            # Update the maximum lengths seen so far.
            max_cache_length = max(max_cache_length, cache_len)

        # Build a tensor of maximum lengths. Each step slices the first row to
        # advance to the values for the next row.
        max_lengths_host = build_max_lengths_tensor(
            num_steps, max_seq_length, max_cache_length
        )

        # This is batch_size x max_num_pages.
        # Note that each page is enough to fit up to vision_max_seq_len so there
        # is at most one page per sequence.

        lookup_table_tensor_vision = Tensor.from_numpy(
            np.array(
                [
                    [
                        self.vision_kv_manager.request_to_seq_id[
                            ctx.request_id
                        ],
                        1,
                    ]
                    for ctx in batch
                ],
                np.uint32,
            )
        )

        vision_fetch_results = RaggedKVCacheInputs(
            # Block 0 for the first device (since MultimodalKVCacheManager
            # assumes only 1 device).
            blocks=self.vision_kv_manager.device_tensors[0],
            cache_lengths=Tensor.from_numpy(cache_lengths_np).to(device),
            lookup_table=lookup_table_tensor_vision.to(device),
            max_lengths=max_lengths_host,
        )

        multimodal_kv_inputs = [
            MultimodalKVCacheInputs(text_fetch_results, vision_fetch_results)
        ]
        return cast(list[KVCacheInputs], multimodal_kv_inputs)

    @final
    def input_symbols(
        self,
    ) -> Sequence[MultimodalKVCacheInputSymbols]:
        """Returns concatenated input symbols for text and vision KV managers.

        This renames symbolic dimensions to avoid conflicts between text and
        vision modalities which may have different numbers of pages/layers.
        """
        # Get input symbols from both managers
        text_symbols = self.text_kv_manager.input_symbols()[0]
        vision_symbols = self.vision_kv_manager.input_symbols()[0]

        # Rename conflicting symbolic dimensions in text symbols
        text_symbols.kv_blocks.shape[0] = SymbolicDim("text_total_num_pages")
        text_symbols.lookup_table.shape[1] = SymbolicDim("text_max_num_pages")

        # Rename conflicting symbolic dimensions in vision symbols
        vision_symbols.kv_blocks.shape[0] = SymbolicDim(
            "vision_total_num_pages"
        )
        vision_symbols.lookup_table.shape[1] = SymbolicDim(
            "vision_max_num_pages"
        )

        # Also rename the num_layers dimension which differs between modalities
        text_symbols.kv_blocks.shape[2] = SymbolicDim("text_num_layers")
        vision_symbols.kv_blocks.shape[2] = SymbolicDim("vision_num_layers")

        return [
            MultimodalKVCacheInputSymbols(
                text_kv_input_symbols=text_symbols,
                vision_kv_input_symbols=vision_symbols,
            )
        ]

    def step(self, batch: list[InputContext]) -> None:
        """Steps both text and vision modalities' KV managers."""
        # Step the text KV manager as usual for autoregressive text generation.
        self.text_kv_manager.step(batch)

        # Keep the base class's state in sync with the text KV manager's.
        super().step(batch)

    def external_claim(self, request_id: RequestID) -> None:
        """Reserves sequence IDs for the given request ID in both modalities' KV caches."""
        self.text_kv_manager.external_claim(request_id)
        self.vision_kv_manager.external_claim(request_id)

        # Keep the base class's state in sync with the text KV manager's.
        super().external_claim(request_id)

    def release(self, request_id: RequestID) -> None:
        """Marks the sequence complete for both modalities' KV caches."""
        self.text_kv_manager.release(request_id)
        self.vision_kv_manager.release(request_id)
        super().release(request_id)

    def contains(self, request_id: RequestID) -> bool:
        """Returns whether `request_id` is in the KV cache."""
        text_kv_contains = self.text_kv_manager.contains(request_id)

        # Assume that the modalities' KV caches have consistent request ids.
        assert text_kv_contains == self.vision_kv_manager.contains(request_id)

        return text_kv_contains

    def num_kv_inputs(self) -> int:
        """Returns the sum of the KV input lengths for both modalities.

        Each KV manager (text and vision) returns 4 inputs, so the total is 8.
        """
        return 8

    def increment_cache_lengths(
        self,
        kv_cache_inputs: list[RaggedKVCacheInputs] | list[PaddedKVCacheInputs],
        prev_model_inputs: Iterable[Any],
    ) -> list[RaggedKVCacheInputs] | list[PaddedKVCacheInputs]:
        """Updates the cache lengths for multistep execution.

        This increments the text and vision KV cache lengths separately using
        their respective KV cache inputs.
        """
        # Cast the input to MultimodalKVCacheInputs to access its components
        multimodal_inputs = cast(list[MultimodalKVCacheInputs], kv_cache_inputs)
        text_kv_inputs = multimodal_inputs[0].text_kv_cache_inputs
        vision_kv_inputs = multimodal_inputs[0].vision_kv_cache_inputs

        multimodal_kv_inputs = [
            MultimodalKVCacheInputs(
                text_kv_cache_inputs=self.text_kv_manager.increment_cache_lengths(
                    [text_kv_inputs],  # type: ignore
                    prev_model_inputs,
                )[0],
                vision_kv_cache_inputs=self.vision_kv_manager.increment_cache_lengths(
                    [vision_kv_inputs],  # type: ignore
                    prev_model_inputs,
                )[0],
            )
        ]
        return cast(list[RaggedKVCacheInputs], multimodal_kv_inputs)


# TODO(bduke): use `@dataclass(slots=True)` when we drop 3.9 support.
class LlamaVisionInputs(ModelInputs):
    """Holds language model inputs and (optionally) vision model inputs."""

    # Language model inputs.
    input_id_values: Tensor
    input_row_offsets: Tensor
    input_id_max_seq_len: Tensor
    pixel_row_offsets: Tensor

    # Vision model inputs.
    _pixel_values: Tensor | None = None
    _aspect_ratio_ids: Tensor | None = None
    _aspect_ratio_mask: Tensor | None = None

    def __init__(
        self,
        input_id_values: Tensor,
        input_row_offsets: Tensor,
        input_id_max_seq_len: Tensor,
        pixel_row_offsets: Tensor,
        pixel_values: Tensor | None = None,
        aspect_ratio_ids: Tensor | None = None,
        aspect_ratio_mask: Tensor | None = None,
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> None:
        self.input_id_values = input_id_values
        self.input_row_offsets = input_row_offsets
        self.input_id_max_seq_len = input_id_max_seq_len
        self.pixel_row_offsets = pixel_row_offsets
        self._pixel_values = pixel_values
        self._aspect_ratio_ids = aspect_ratio_ids
        self._aspect_ratio_mask = aspect_ratio_mask
        self.kv_cache_inputs = kv_cache_inputs

    def __post_init__(self) -> None:
        """Validate consistency between vision fields.

        If pixel_values is set, then aspect_ratio_ids, aspect_ratio_mask,
        and pixel_row_offsets must also be set, and vice versa.
        """
        if self.has_vision_inputs:
            if not all(
                x is not None
                for x in (
                    self._aspect_ratio_ids,
                    self._aspect_ratio_mask,
                    self.pixel_row_offsets,
                )
            ):
                msg = "provide all or none of Llama Vision vision model inputs"
                raise ValueError(msg)
        else:
            for field_name in ("_aspect_ratio_ids", "_aspect_ratio_mask"):
                if getattr(self, field_name) is not None:
                    msg = f"{field_name} must be None if _pixel_values is None"
                    raise ValueError(msg)

    @property
    def has_vision_inputs(self) -> bool:
        """Returns true iff this includes vision model inputs."""
        return self._pixel_values is not None

    @property
    def pixel_values(self) -> Tensor:
        assert self._pixel_values is not None
        return self._pixel_values

    @property
    def aspect_ratio_ids(self) -> Tensor:
        assert self._aspect_ratio_ids is not None
        return self._aspect_ratio_ids

    @property
    def aspect_ratio_mask(self) -> Tensor:
        assert self._aspect_ratio_mask is not None
        return self._aspect_ratio_mask

    def update_for_next_token(
        self,
        next_tokens: Tensor,
        next_row_offsets: Tensor,
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> LlamaVisionInputs:
        """Updates next_tokens and row_offsets after an initial step."""
        return LlamaVisionInputs(
            input_id_values=next_tokens,
            input_row_offsets=next_row_offsets,
            input_id_max_seq_len=self.input_id_max_seq_len,
            pixel_row_offsets=self.pixel_row_offsets,
            # Set vision model inputs to None after the first `next_token`.
            pixel_values=None,
            aspect_ratio_ids=None,
            aspect_ratio_mask=None,
            kv_cache_inputs=kv_cache_inputs,
        )


class LlamaVisionModel(Layer):
    """The Llama 3.2 vision model."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        weights: Weights,
        huggingface_config: AutoConfig,
        dtype: DType,
    ) -> None:
        # Set convenience attributes for the text and vision configs.
        self.vision_config = huggingface_config.vision_config
        self.text_config = huggingface_config.text_config

        self.vision_model = instantiate_vision_model(
            dtype=dtype,
            image_size=self.vision_config.image_size,
            patch_size=self.vision_config.patch_size,
            supported_aspect_ratios=self.vision_config.supported_aspect_ratios,
            hidden_size=self.vision_config.hidden_size,
            max_num_tiles=self.vision_config.max_num_tiles,
            num_channels=self.vision_config.num_channels,
            norm_eps=self.vision_config.norm_eps,
            attention_heads=self.vision_config.attention_heads,
            num_hidden_layers=self.vision_config.num_hidden_layers,
            intermediate_size=self.vision_config.intermediate_size,
            num_global_layers=self.vision_config.num_global_layers,
            intermediate_layers_indices=self.vision_config.intermediate_layers_indices,
            weights=weights,
            device=DeviceRef.GPU(),
        )

        self.multi_modal_projector = LinearV1(
            weights.multi_modal_projector.weight.allocate(
                dtype,
                [
                    self.text_config.hidden_size,
                    self.vision_config.vision_output_dim,
                ],
                device=DeviceRef.GPU(),
            ),
            weights.multi_modal_projector.bias.allocate(
                dtype, [self.text_config.hidden_size], device=DeviceRef.GPU()
            ),
        )

    def __call__(
        self,
        pixel_values: TensorValue,
        aspect_ratio_ids: TensorValue,
        aspect_ratio_mask: TensorValue,
    ) -> TensorValue:
        if aspect_ratio_ids is None:
            msg = "`aspect_ratio_ids` must be provided if `pixel_values` is provided"
            raise ValueError(msg)

        # Get vision tokens from vision model.
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
        )
        cross_attention_states = vision_outputs[0]

        num_patches = cross_attention_states.shape[-2]

        return self.multi_modal_projector(cross_attention_states).reshape(
            [
                Dim("batch_size")
                * Dim("num_concurrent_media")
                * self.vision_config.max_num_tiles
                * num_patches,
                self.text_config.hidden_size,
            ]
        )


class LlamaVisionLanguageModel(Layer):
    """The Llama 3.2 vision language model."""

    language_model: CausalLanguageModel
    """Language model composed of self and cross attention layers."""

    num_text_kv_cache_inputs: int
    """Number of KV cache inputs for self attention layers."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        weights: Weights,
        kv_params: KVCacheParams,
        vision_kv_params: KVCacheParams,
        max_seq_len: int,
        num_text_kv_cache_inputs: int,
        huggingface_config: AutoConfig,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        text_config = huggingface_config.text_config
        self.language_model = instantiate_language_model(
            dtype=dtype,
            hidden_size=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            rope_theta=text_config.rope_theta,
            max_seq_len=max_seq_len,
            num_hidden_layers=text_config.num_hidden_layers,
            cross_attention_layers=text_config.cross_attention_layers,
            vocab_size=text_config.vocab_size,
            rms_norm_eps=text_config.rms_norm_eps,
            num_key_value_heads=text_config.num_key_value_heads,
            intermediate_size=text_config.intermediate_size,
            kv_params=kv_params,
            vision_kv_params=vision_kv_params,
            weights=weights,
            device=device,
        )
        self.num_text_kv_cache_inputs = num_text_kv_cache_inputs

    def __call__(
        self,
        cross_attention_states: TensorValue,
        input_ids: TensorValue,
        hidden_input_row_offsets: TensorValue,
        hidden_max_seq_len: TensorValue,
        cross_input_row_offsets: TensorValue,
        *kv_cache_inputs: TensorValue,
    ) -> TensorValue:
        logits = self.language_model(
            text_kv_cache_inputs=kv_cache_inputs[
                : self.num_text_kv_cache_inputs
            ],
            vision_kv_cache_inputs=kv_cache_inputs[
                self.num_text_kv_cache_inputs :
            ],
            input_ids=input_ids,
            hidden_input_row_offsets=hidden_input_row_offsets,
            hidden_max_seq_len=hidden_max_seq_len,
            cross_attention_states=cross_attention_states,
            cross_input_row_offsets=cross_input_row_offsets,
        )

        # Always return float32 logits, no matter the activation type
        return ops.cast(logits, DType.float32)


class LlamaVision(PipelineModel[TextAndVisionContext]):  # type: ignore
    """The entire (multimodal) Llama3.2 vision model.

    A note on multi-step and vision inputs:

    - `has_images` in `prepare_initial_token_inputs` is determined by whether or
      not `pixel_values` is set on each TextAndVisionContext in the batch.
      So on the context encoding call, the caller sets pixel_values, making
      `has_images` True.
    - `prepare_initial_token_inputs` unsets `ctx.pixel_values` (sets it to an
      empty list).
      So the next prepare_initial_token_inputs will have has_images == False
      (the next multi-step train will skip the vision encoder).
    - That covers the num_steps = 1 case.
    - For multistep, the prepare_next_token_inputs function will unset
      LlamaVisionInputs.pixel_values (and aspect ratio ids/mask).
      So for multistep, step > 1, subsequent steps won't run the vision encoder.
    - Note the 2 different mechanisms: `has_images` is determined by
      `TextAndVisionContext.pixel_values` in `prepare_initial_token_inputs`,
      but it is determined by `LlamaVisionInputs.pixel_values` in
      `PipelineModel.execute` (which is called multiple times in a multi-step
      train, so `prepare_next_token_inputs` needs to unset
      `LlamaVisionInputs.pixel_values`).
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: Optional[WeightsAdapter] = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        # Set convenience attributes for the text and vision configs.
        self.vision_config = huggingface_config.vision_config
        self.text_config = huggingface_config.text_config

        # These need to be set at graph instantiation time.
        self.vision_graph_input_size = -1
        self.language_graph_input_size = -1

        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )
        self.vision_model, self.language_model = self.load_model(session)
        # Note that in a multimodal model, the language model is the last model in the
        # pipeline. Unfortunately, self.model is still being used (and exposed)
        # in the token generation code, so we still need to set it here.
        self.model = self.language_model

    def _llama3_vision_vision_graph(self) -> Graph:
        # NOTE: Llama 3.2 vision only supports single-device, currently.
        device = DeviceRef(self.devices[0].label, self.devices[0].id)

        # Inserted a manual CHW -> HWC transpose here.
        pixel_values_type = TensorType(
            # This has to be of type float32 as we construct tensors from a numpy
            # array (which has no notion of some dtypes like bfloat16). Explicit
            # casting will happen inside the graph.
            DType.float32,
            shape=[
                "batch_size",
                "num_concurrent_media",
                self.vision_config.max_num_tiles,
                self.vision_config.image_size,  # height
                self.vision_config.image_size,  # width
                self.vision_config.num_channels,
            ],
            device=device,
        )
        aspect_ratio_ids_type = TensorType(
            DType.int64,
            shape=["batch_size", "num_concurrent_media"],
            device=device,
        )
        aspect_ratio_mask_type = TensorType(
            DType.int64,
            shape=[
                "batch_size",
                "num_concurrent_media",
                self.vision_config.max_num_tiles,
            ],
            device=device,
        )

        input_types = [
            pixel_values_type,
            aspect_ratio_ids_type,
            aspect_ratio_mask_type,
        ]
        self.vision_graph_input_size = len(input_types)
        return Graph(
            "llama3-vision-vision-model-graph",
            forward=LlamaVisionModel(
                pipeline_config=self.pipeline_config,
                weights=self.weights,
                huggingface_config=self.huggingface_config,
                dtype=self.dtype,
            ),
            input_types=input_types,
        )

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.text_config.max_position_embeddings,
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            msg = (
                "Unable to infer max_length for Llama Vision, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.text_config.max_position_embeddings})."
            )
            raise ValueError(msg) from e

    def _llama3_vision_language_graph(self) -> Graph:
        # Pre-allocate a buffer for input_row_offsets in multistep execution.
        # We do this to avoid materializing and copying a buffer with each multistep step
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        # NOTE: Llama 3.2 vision only supports single-device, currently.
        device = DeviceRef(self.devices[0].label, self.devices[0].id)
        input_ids_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device
        )
        # image_size = self.vision_config.image_size
        # patch_size = self.vision_config.patch_size
        cross_attention_states_type = TensorType(
            self.dtype,
            shape=[
                # TODO(bduke): fix algebraic dim creation outside of graph
                # contexts.
                # Dim("batch_size")
                # * "num_concurrent_media"
                # * self.vision_config.max_num_tiles
                # * ((image_size // patch_size) ** 2 + 1),
                "num_vision_embeddings",
                self.text_config.hidden_size,
            ],
            device=device,
        )
        input_ids_max_seq_len_type = TensorType(
            DType.uint32, [1], device=DeviceRef.CPU()
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device
        )
        cross_row_offsets_type = input_row_offsets_type

        # Unpack multimodal KV inputs.
        assert isinstance(self.kv_manager, MultimodalKVCacheManager)
        input_symbols = self.kv_manager.input_symbols()[0]
        text_kv_input_symbols = input_symbols.text_kv_input_symbols
        vision_kv_input_symbols = input_symbols.vision_kv_input_symbols

        input_types = [
            cross_attention_states_type,
            input_ids_type,
            input_row_offsets_type,
            input_ids_max_seq_len_type,
            cross_row_offsets_type,
            *text_kv_input_symbols,
            *vision_kv_input_symbols,
        ]
        self.language_graph_input_size = len(input_types)

        kv_params = self.kv_manager.params
        vision_kv_params = self.kv_manager.vision_kv_params

        return Graph(
            "llama3-vision-language-model-graph",
            forward=LlamaVisionLanguageModel(
                pipeline_config=self.pipeline_config,
                weights=self.weights,
                kv_params=kv_params,
                vision_kv_params=vision_kv_params,
                max_seq_len=self.calculate_max_seq_len(
                    self.pipeline_config,
                    huggingface_config=self.huggingface_config,
                ),
                num_text_kv_cache_inputs=len(list(text_kv_input_symbols)),
                huggingface_config=self.huggingface_config,
                dtype=self.dtype,
                device=device,
            ),
            input_types=input_types,
        )

    @property
    def vision_max_seq_len(self) -> int:
        """Returns the maximum number of vision tokens."""
        return self._calculate_vision_max_seq_len(self.huggingface_config)

    @classmethod
    def _calculate_vision_max_seq_len(
        cls, huggingface_config: AutoConfig
    ) -> int:
        """Returns the maximum number of vision tokens."""
        # Marshal out hyperparameters.
        height = huggingface_config.vision_config.image_size
        width = huggingface_config.vision_config.image_size
        max_num_tiles = huggingface_config.vision_config.max_num_tiles
        patch_size = huggingface_config.vision_config.patch_size
        # TODO(bduke): account for the actual instead of max number of tiles.
        # num_tiles * (image_dim**2 // patch_dim**2 + 1 (cls token))
        return max_num_tiles * ((height * width) // patch_size**2 + 1)

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return LlamaVisionConfig.get_num_layers(huggingface_config)

    def _prepare_vision_inputs(
        self,
        context_batch: Sequence[TextAndVisionContext],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Batches up pixel_values, aspect_ratio_ids, and aspect_ratio_masks."""
        images = []
        aspect_ratio_ids_list: list[np.ndarray] = []
        aspect_ratio_mask_list: list[np.ndarray] = []
        for context in context_batch:
            # Get first image in first batch and permute the order to (HWC).
            image = np.transpose(context.pixel_values[0], (0, 1, 3, 4, 2))

            # Add batch_size, num_concurrent_media, and max_num_tiles dimensions
            # [1, num_concurrent_media, max_num_tiles, H, W, C]
            image = np.expand_dims(image, axis=(0))
            images.append(image)

            if "aspect_ratio_ids" not in context.extra_model_args:
                msg = "aspect_ratio_ids is required for image / vision model input"
                raise ValueError(msg)

            if "aspect_ratio_mask" not in context.extra_model_args:
                msg = "aspect_ratio_mask is required for image / vision model input"
                raise ValueError(msg)

            aspect_ratio_ids_list.append(
                context.extra_model_args["aspect_ratio_ids"]
            )
            aspect_ratio_mask_list.append(
                context.extra_model_args["aspect_ratio_mask"]
            )

        # Convert the list into a single NumPy array with shape
        # (batch_size, 1, max_num_tiles, H, W, C).
        final_images = np.concatenate(images, axis=0)

        pixel_values = Tensor.from_numpy(final_images).to(self.devices[0])

        final_aspect_ratio_ids = np.concatenate(aspect_ratio_ids_list, axis=0)

        aspect_ratio_ids = Tensor.from_numpy(final_aspect_ratio_ids).to(
            self.devices[0]
        )

        final_aspect_ratio_mask = np.concatenate(aspect_ratio_mask_list, axis=0)

        aspect_ratio_mask = Tensor.from_numpy(final_aspect_ratio_mask).to(
            self.devices[0]
        )

        return pixel_values, aspect_ratio_ids, aspect_ratio_mask

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextAndVisionContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> LlamaVisionInputs:
        """Creates tensors of token and image inputs, if applicable."""
        if self.kv_cache_config.cache_strategy != KVCacheStrategy.PAGED:
            msg = "Llama Vision only supports paged cache strategy"
            raise ValueError(msg)

        has_images = any(ctx.needs_vision_encoding for ctx in context_batch)
        if has_images and not all(
            ctx.needs_vision_encoding for ctx in context_batch
        ):
            msg = (
                "expected context batch to all have images, or no images at all"
            )
            raise RuntimeError(msg)

        def initial_prompt_missing_image(ctx: TextAndVisionContext) -> bool:
            return ctx.is_initial_prompt and not ctx.pixel_values

        if any(initial_prompt_missing_image(ctx) for ctx in context_batch):
            msg = "The Llama Vision model currently requires a prompt with an image. Consider using the regular text-only models for non-image prompts"
            raise RuntimeError(msg)

        # Prepare vision inputs if applicable.
        pixel_values = None
        aspect_ratio_ids = None
        aspect_ratio_mask = None
        if has_images:
            pixel_values, aspect_ratio_ids, aspect_ratio_mask = (
                self._prepare_vision_inputs(context_batch)
            )

        # Input row offset type: ["input_row_offsets_len"], UInt32
        input_id_row_offsets = Tensor.from_numpy(
            np.cumsum(
                [0] + [ctx.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        pixel_row_offsets = Tensor.from_numpy(
            np.cumsum(
                [0]
                + [
                    # Use an input row offset of 0 to mean no image.
                    self.vision_max_seq_len if ctx.needs_vision_encoding else 0
                    for ctx in context_batch
                ],
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        # Input Ids: ["total_seq_len"], Int64
        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])
        input_id_values = Tensor.from_numpy(tokens).to(self.devices[0])

        # This lives on host here and in the kernel.
        input_id_max_seq_len = Tensor.from_numpy(
            np.array(
                [max(ctx.active_length for ctx in context_batch)],
                dtype=np.uint32,
            )
        )

        # Mark that vision encoding is complete for all contexts in the batch
        # This prevents re-encoding on subsequent calls before update() is called
        for ctx in context_batch:
            ctx.needs_vision_encoding = False

        return LlamaVisionInputs(
            input_id_values=input_id_values,
            input_row_offsets=input_id_row_offsets,
            input_id_max_seq_len=input_id_max_seq_len,
            pixel_row_offsets=pixel_row_offsets,
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_inputs: ModelInputs,
    ) -> LlamaVisionInputs:
        """Produce the updated LlamaVisionInputs for the next token.

        This sets existing vision inputs to none and replaces text tokens and
        row offsets.
        """
        prev_inputs = cast(LlamaVisionInputs, prev_inputs)
        next_row_offsets = self._input_row_offsets_prealloc[
            : prev_inputs.input_row_offsets.shape[0]
        ]

        return prev_inputs.update_for_next_token(
            next_tokens, next_row_offsets, prev_inputs.kv_cache_inputs
        )

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        assert model_inputs.kv_cache_inputs is not None, (
            "Llama Vision has KV cache inputs"
        )
        # batch_size * num_concurrent_media * max_num_tiles * num_patches
        # are set to 0 here to imitate a dummy tensor (used in text-only mode).
        cross_attention_states = Tensor.zeros(
            shape=[0, self.text_config.hidden_size], dtype=self.dtype
        ).to(self.devices[0])

        model_inputs = cast(LlamaVisionInputs, model_inputs)
        if model_inputs.has_vision_inputs:
            # Compute the cross attention states if this is a CE step.
            exec_result = self.vision_model.execute(
                model_inputs.pixel_values,
                model_inputs.aspect_ratio_ids,
                model_inputs.aspect_ratio_mask,
            )[0]
            assert isinstance(exec_result, Tensor)
            cross_attention_states = exec_result

        all_kv_cache_inputs: list[Tensor] = []
        if isinstance(model_inputs.kv_cache_inputs, MultimodalKVCacheInputs):
            all_kv_cache_inputs.extend(
                model_inputs.kv_cache_inputs.text_kv_cache_inputs
            )
            all_kv_cache_inputs.extend(
                model_inputs.kv_cache_inputs.vision_kv_cache_inputs
            )
        elif isinstance(model_inputs.kv_cache_inputs, KVCacheInputs):
            all_kv_cache_inputs = list(model_inputs.kv_cache_inputs)
        else:
            raise ValueError(
                f"Unsupported kv_cache_inputs type: {type(model_inputs.kv_cache_inputs)}"
            )

        model_outputs = self.language_model.execute(
            cross_attention_states,
            model_inputs.input_id_values,
            model_inputs.input_row_offsets,
            model_inputs.input_id_max_seq_len,
            model_inputs.pixel_row_offsets,
            *all_kv_cache_inputs,
        )
        if len(model_outputs) == 3:
            return ModelOutputs(
                next_token_logits=cast(Tensor, model_outputs[0]),
                logits=cast(Tensor, model_outputs[1]),
                logit_offsets=cast(Tensor, model_outputs[2]),
            )
        else:
            return ModelOutputs(
                next_token_logits=cast(Tensor, model_outputs[0]),
                logits=cast(Tensor, model_outputs[0]),
            )

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return LlamaVisionConfig.get_kv_params(
            huggingface_config=huggingface_config,
            n_devices=n_devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
        """Loads KV cache management objects for Llama vision.

        Args:
            session: Inference session to compile and init the KV cache.
            available_cache_memory: Amount of memory available to the KV cache,
                in bytes.

        Returns:
            A pair of KV managers: one for self the other for cross attention.
        """
        num_cross_attn_layers = len(self.text_config.cross_attention_layers)
        return MultimodalKVCacheManager(
            params=self.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            text_max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            vision_max_seq_len=self.vision_max_seq_len,
            text_num_layers=self.text_config.num_hidden_layers
            - num_cross_attn_layers,
            vision_num_layers=num_cross_attn_layers,
            devices=self.devices,
            session=session,
            available_cache_memory=available_cache_memory,
            page_size=self.kv_cache_config.kv_cache_page_size,
        )

    @classmethod
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        devices: list[Device],
        huggingface_config: AutoConfig,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        """Estimates the size of the kv cache in bytes."""
        assert pipeline_config.max_batch_size is not None

        num_cross_attn_layers = len(
            huggingface_config.text_config.cross_attention_layers
        )
        return MultimodalKVCacheManager.estimated_memory_size(
            params=cls.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=len(devices),
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            num_layers=huggingface_config.text_config.num_hidden_layers
            - num_cross_attn_layers,
            available_cache_memory=available_cache_memory,
            devices=devices,
            max_vision_seq_len=cls._calculate_vision_max_seq_len(
                huggingface_config
            ),
            num_vision_layers=num_cross_attn_layers,
        )

    @classmethod
    def infer_optimal_batch_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        huggingface_config: AutoConfig,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        if len(devices) == 1 and devices[0].is_host:
            return 1

        num_cross_attn_layers = len(
            huggingface_config.text_config.cross_attention_layers
        )
        optimal_batch_size = MultimodalKVCacheManager.infer_optimal_batch_size(
            params=cls.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=len(devices),
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            num_layers=huggingface_config.text_config.num_hidden_layers
            - num_cross_attn_layers,
            # TODO(GEX-1843): we underestimate the memory usage of the
            # vision model due to multiple activations in flight while executing the
            # vision encoder and first few layers of the text encoder in parallel.
            # This is a hacky workaround to account for this, in the long term we
            # should more accurately measure a model's memory consumption via
            # an interface in the graph compiler.
            available_cache_memory=int(available_cache_memory * 0.8),
            devices=devices,
            max_vision_seq_len=cls._calculate_vision_max_seq_len(
                huggingface_config
            ),
            num_vision_layers=num_cross_attn_layers,
        )
        return max(
            cls._MIN_DEFAULT_BATCH_SIZE,
            min(optimal_batch_size, cls._MAX_DEFAULT_BATCH_SIZE),
        )

    def load_model(
        self,
        session: InferenceSession,
    ) -> tuple[Model, Model]:
        """
        Load the Llama vision multimodal model. Since this is a multimodal model,
        we have vision and language models (graph) loaded.
        """

        def build_vision_model():
            logger.info("Building and compiling vision model...")
            before = time.perf_counter()
            vision_model_graph = self._llama3_vision_vision_graph()
            vision_model = session.load(
                vision_model_graph,
                weights_registry=self.weights.allocated_weights,
            )
            after = time.perf_counter()
            logger.info(
                f"Compiling vision model took {after - before:.6f} seconds"
            )
            return vision_model

        def build_language_model():
            logger.info("Building and compiling language model...")
            before = time.perf_counter()
            language_model_graph = self._llama3_vision_language_graph()
            language_model = session.load(
                language_model_graph,
                weights_registry=self.weights.allocated_weights,
            )
            after = time.perf_counter()
            logger.info(
                f"Building and compiling language model took {after - before:.6f} seconds"
            )
            return language_model

        if _DO_PARALLEL_COMPILATION:
            with ThreadPoolExecutor(max_workers=2) as executor:
                vision_model_future = executor.submit(build_vision_model)
                language_model_future = executor.submit(build_language_model)
                vision_model = vision_model_future.result()
                language_model = language_model_future.result()
        else:
            vision_model = build_vision_model()
            language_model = build_language_model()

        return (vision_model, language_model)
