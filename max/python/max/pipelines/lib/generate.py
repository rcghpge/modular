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

"""Fixed-batch generation, the lowest-level entry point to the pipeline API.

This runs a pipeline's forward passes over a batch you choose, with nothing
added on top of the pipeline API. The caller's list of prompts is the batch:
it is never split, re-packed, or chunked, and every :meth:`Batch.step` runs
exactly one ``pipeline.execute`` in the caller's thread. That also makes it
useful for testing, such as reproducing and localizing kernel and model
accuracy bugs. Serving features (scheduling and in-flight batching, chunked
prefill, prefix caching, LoRA, and structured output) are tested through
``max serve``. Kernels are not batch-invariant, so for bit-exact comparisons
between MAX runs, use batch size 1 or ``ignore_eos=True``.
"""

from __future__ import annotations

import dataclasses
import logging
import operator
import shlex
import weakref
from collections.abc import Iterable, Iterator, Mapping, Sequence
from types import TracebackType
from typing import Protocol, runtime_checkable

import numpy as np
from max.driver import DeviceSpec
from max.pipelines.context import (
    EOSTracker,
    SamplingParams,
    SamplingParamsInput,
    TextAndVisionContext,
    TextContext,
    TextGenerationOutput,
    TokenBuffer,
)
from max.pipelines.kv_cache.kv_connector import KVTransfer
from max.pipelines.modeling.types import (
    PipelineOutputsDict,
    RequestID,
    TextGenerationInputs,
)

from .config import PipelineConfig
from .registry import PIPELINE_REGISTRY

__all__ = [
    "Batch",
    "BatchPipeline",
    "GenerateResult",
    "generate",
]

# A row's Philox key is its seed plus its token count, so rows seeded this
# far apart never share a key while every row is under 2**32 tokens.
_ROW_SEED_STRIDE = 1 << 32
_U64_MASK = (1 << 64) - 1

_logger = logging.getLogger("max.pipelines")


class _KVCacheParams(Protocol):
    @property
    def page_size(self) -> int: ...


class _KVCacheClaims(Protocol):
    @property
    def params(self) -> _KVCacheParams: ...

    def claim(self, ctx: TextContext, replica_idx: int = 0) -> None: ...

    def alloc(self, ctx: TextContext) -> KVTransfer: ...

    def release(self, ctx: TextContext) -> None: ...

    def contains(self, ctx: TextContext) -> bool: ...


class _EOSTokenSource(Protocol):
    @property
    def eos_token_ids(self) -> set[int]: ...


# Only text tokenizers know their vocab size. PipelineTokenizer omits it.
@runtime_checkable
class _VocabSizeSource(Protocol):
    @property
    def tokenizer_vocab_size(self) -> int: ...


@runtime_checkable
class _OverlapSource(Protocol):
    @property
    def overlap_active(self) -> bool: ...


@runtime_checkable
class _GraphCaptureSource(Protocol):
    @property
    def graph_capture_ready(self) -> bool: ...


class _BatchProcessorRuntime(Protocol):
    @property
    def max_batch_active_tokens(self) -> int: ...


class _BatchProcessor(Protocol):
    @property
    def runtime(self) -> _BatchProcessorRuntime: ...


class _PipelineModelLimits(Protocol):
    @property
    def max_seq_len(self) -> int: ...

    @property
    def batch_processor(self) -> _BatchProcessor | None: ...

    @property
    def planned_max_batch_total_tokens(self) -> int | None: ...


class BatchPipeline(Protocol):
    """The pipeline surface :class:`Batch` drives.

    :class:`~max.pipelines.lib.TextGenerationPipeline` and
    :class:`~max.pipelines.lib.OverlapTextGenerationPipeline` over
    :class:`~max.pipelines.context.TextContext` satisfy it. A pipeline whose
    ``execute`` returns the previous call's results also defines a boolean
    ``overlap_active`` property, read the way serve's schedulers read it. A
    pipeline that can replay captured device graphs defines a boolean
    ``graph_capture_ready`` property, true once its graphs are captured.
    """

    @property
    def kv_manager(self) -> _KVCacheClaims:
        """The KV cache manager that owns each row's cache blocks."""
        ...

    @property
    def pipeline_config(self) -> PipelineConfig:
        """The config supplying the DP degree and sampling defaults."""
        ...

    @property
    def tokenizer(self) -> _EOSTokenSource:
        """The tokenizer supplying the model's EOS token ids."""
        ...

    @property
    def max_batch_size(self) -> int:
        """The planned maximum batch size per data-parallel replica."""
        ...

    # The memory plan's max length and batch token limits live only on the
    # model.
    @property
    def _pipeline_model(self) -> _PipelineModelLimits: ...

    def execute(
        self, inputs: TextGenerationInputs[TextContext]
    ) -> PipelineOutputsDict[TextGenerationOutput]:
        """Runs one model step over ``inputs``."""
        ...

    def release(self, request_id: RequestID) -> None:
        """Releases the pipeline's per-request state for ``request_id``."""
        ...


@dataclasses.dataclass(frozen=True)
class GenerateResult:
    """The result of :func:`generate`."""

    sequences: list[list[int]]
    """Each row's prompt followed by its new tokens, in input order, as in
    Hugging Face ``generate``."""


# A pipeline that is not synchronous hands its pending forward's results to
# whichever batch steps next, so two open batches would trade outputs.
_open_batches: weakref.WeakKeyDictionary[object, weakref.ref[Batch]] = (
    weakref.WeakKeyDictionary()
)


def _int_rows(values: Iterable[Sequence[int]], name: str) -> list[list[int]]:
    try:
        return [[operator.index(value) for value in row] for row in values]
    except TypeError:
        raise TypeError(
            f"{name} must be a list of integer lists, one per prompt. Wrap "
            "one prompt as [ids]."
        ) from None


def _materialize_prompts(
    input_ids: Iterable[Sequence[int]],
    attention_mask: Iterable[Sequence[int]] | None,
) -> list[list[int]]:
    prompts = _int_rows(input_ids, "input_ids")
    if attention_mask is None:
        return prompts
    masks = _int_rows(attention_mask, "attention_mask")
    if [len(mask) for mask in masks] != [len(prompt) for prompt in prompts]:
        raise ValueError(
            "attention_mask must have the same shape as input_ids."
        )
    # Batch runs prompts unpadded, so it drops the positions a zero masks.
    return [
        [token for token, keep in zip(prompt, mask, strict=True) if keep]
        for prompt, mask in zip(prompts, masks, strict=True)
    ]


def _resolve_sampling_params(
    pipeline: BatchPipeline, requested: SamplingParamsInput
) -> SamplingParams:
    # Unset knobs fall back to the server-level defaults before the
    # generation config, as serve resolves a request's sampling params.
    runtime = pipeline.pipeline_config.runtime
    return SamplingParams.from_input_and_generation_config(
        dataclasses.replace(
            requested,
            temperature=(
                requested.temperature
                if requested.temperature is not None
                else runtime.temperature
            ),
            top_k=(
                requested.top_k
                if requested.top_k is not None
                else runtime.top_k
            ),
        ),
        sampling_params_defaults=pipeline.pipeline_config.model.sampling_params_defaults,
    )


def _build_contexts(
    pipeline: BatchPipeline,
    prompts: Sequence[Sequence[int]],
    *,
    max_new_tokens: int,
    sampling_params: SamplingParams,
) -> list[TextContext]:
    """Builds one context per prompt."""
    planned_max_length = pipeline._pipeline_model.max_seq_len
    if sampling_params.min_new_tokens > max_new_tokens:
        raise ValueError(
            f"min_new_tokens={sampling_params.min_new_tokens}, from the "
            "arguments or the generation config, exceeds "
            f"max_new_tokens={max_new_tokens}. Raise max_new_tokens."
        )
    tokenizer = pipeline.tokenizer
    # The same set serve's tokenizer builds for a request.
    eos_token_ids: set[int] = set()
    if not sampling_params.ignore_eos:
        eos_token_ids.update(tokenizer.eos_token_ids)
        eos_token_ids.update(sampling_params.stop_token_ids or ())
    vocab_size = (
        tokenizer.tokenizer_vocab_size
        if isinstance(tokenizer, _VocabSizeSource)
        else None
    )

    min_new_tokens = max(sampling_params.min_new_tokens, 1)
    contexts = []
    for row, prompt in enumerate(prompts):
        if len(prompt) + min_new_tokens > planned_max_length:
            raise ValueError(
                f"Row {row} needs at least {len(prompt) + min_new_tokens} "
                f"tokens ({len(prompt)} prompt + {min_new_tokens} new), but "
                f"the pipeline's planned max length is {planned_max_length}. "
                "Raise max_length."
            )
        max_length = len(prompt) + max_new_tokens
        if sampling_params.ignore_eos and max_length > planned_max_length:
            raise ValueError(
                f"Row {row} needs {max_length} tokens, but the pipeline's "
                f"planned max length is {planned_max_length}, and with "
                "ignore_eos every row runs all max_new_tokens steps. Raise "
                "max_length or lower max_new_tokens."
            )
        contexts.append(
            TextContext(
                request_id=RequestID(),
                max_length=min(planned_max_length, max_length),
                tokens=TokenBuffer(np.asarray(prompt, dtype=np.int64)),
                eos_tracker=EOSTracker(eos_token_ids=set(eos_token_ids)),
                sampling_params=dataclasses.replace(
                    sampling_params,
                    seed=(sampling_params.seed + row * _ROW_SEED_STRIDE)
                    & _U64_MASK,
                ),
                vocab_size=vocab_size,
            )
        )
    return contexts


def _pipeline_serve_flags(pipeline_config: PipelineConfig) -> list[str]:
    """Returns the ``max serve`` flags that rebuild this pipeline's model.

    A flag is left out when the CLI default gives the same value on any
    GPU host.
    """
    model = pipeline_config.model
    flags: list[str] = []
    if model.huggingface_weight_repo_id != model.model_path:
        # Only a bare repo id passed as the weight path makes discovery read
        # files from another repo.
        flags += ["--weight-path", model.huggingface_weight_repo_id]
    if model.trust_remote_code:
        flags.append("--trust-remote-code")
    devices = model.device_specs
    # Every GPU host defaults to GPU 0. CPU is spelled out because a GPU
    # host would not pick it.
    if devices and devices != [DeviceSpec.accelerator(0)]:
        if any(device.device_type == "cpu" for device in devices):
            flags += ["--devices", "cpu"]
        else:
            gpu_ids = ",".join(str(device.id) for device in devices)
            flags += ["--devices", f"gpu:{gpu_ids}"]
    if model.quantization_encoding is not None:
        flags += ["--quantization-encoding", model.quantization_encoding]
    if model.data_parallel_degree != 1:
        flags += ["--data-parallel-degree", str(model.data_parallel_degree)]
    # Memory planning never shrinks an explicit --max-length, so repeating
    # an architecture default would change the plan.
    if model.max_length_is_user_provided and model.max_length is not None:
        flags += ["--max-length", str(model.max_length)]
    return flags


def _unsupported(
    pipeline_config: PipelineConfig,
    feature: str,
    reason: str,
    serve_flags: Sequence[str] = (),
    *,
    caller: str = "generate()",
    note: str | None = None,
) -> ValueError:
    """Returns the error for a feature ``caller`` leaves to ``max serve``.

    The message names ``caller`` and ``feature``, gives ``reason``, and ends
    with a ``max serve`` command that rebuilds this pipeline with
    ``serve_flags`` added, then ``note`` if given.
    """
    command = shlex.join(
        [
            "max",
            "serve",
            "--model-path",
            pipeline_config.model.model_path,
            *_pipeline_serve_flags(pipeline_config),
            *serve_flags,
        ]
    )
    message = (
        f"{caller} does not support {feature}. {reason}.\n"
        f"{caller} runs the pipeline directly, outside max serve. "
        f"Test {feature} with this max serve command instead.\n  {command}"
    )
    if note is not None:
        message += f"\n{note}"
    return ValueError(message)


def _check_supported(
    pipeline_config: PipelineConfig, caller: str = "generate()"
) -> None:
    """Raises if ``caller`` cannot run the pipeline's model.

    The error names ``caller``, the class the user built. A feature only
    serve runs gets the ``max serve`` command to use instead.
    """
    context_type = PIPELINE_REGISTRY.retrieve_context_type(pipeline_config)
    # TODO(katecaldwell): Support image inputs and image generation, the
    # first follow-up once this API lands.
    if issubclass(context_type, TextAndVisionContext):
        raise ValueError(
            f"{caller} does not support image inputs yet. The model needs "
            f"{context_type.__name__} inputs, and {caller} builds text-only "
            "rows from token ids."
        )
    if not issubclass(context_type, TextContext):
        raise ValueError(
            f"{caller} runs text generation models only, and this model "
            f"uses {context_type.__name__}. Image generation and other model "
            "types are not supported yet."
        )
    lora = pipeline_config.lora
    if lora is not None and lora.enable_lora:
        flags = ["--enable-lora", "--no-enable-prefix-caching"]
        for path in lora.lora_paths:
            flags += ["--lora-paths", path]
        flags += [
            "--max-lora-rank",
            str(lora.max_lora_rank),
            "--max-num-loras",
            str(lora.max_num_loras),
        ]
        if lora.lora_paths:
            # The same naming rule serve's LoRAQueue applies.
            names = ", ".join(
                f'"{path.partition("=")[0]}"' for path in lora.lora_paths
            )
            note = (
                'Then send requests with "model": "<adapter name>". An '
                "adapter's name is the part of its --lora-paths entry before "
                f'"=", or the whole entry if it has no "=" (names here: '
                f"{names})."
            )
        else:
            note = (
                "Load an adapter with POST /v1/load_lora_adapter, then send "
                'requests with "model": "<lora_name>".'
            )
        raise _unsupported(
            pipeline_config,
            "LoRA adapters",
            f"{caller} runs only the base model and never activates an adapter",
            flags,
            caller=caller,
            note=note,
        )


def _check_graph_capture(pipeline: BatchPipeline, dp_degree: int) -> None:
    """Checks Batch against the pipeline's graph capture state.

    Raises if replay needs replica padding, and warns if capture is on but
    the pipeline is not warmed up. Reads whether the pipeline captured its
    graphs, not the config flag, so a pipeline warmed up by hand gets the
    same check.
    """
    pipeline_config = pipeline.pipeline_config
    ready = (
        isinstance(pipeline, _GraphCaptureSource)
        and pipeline.graph_capture_ready
    )
    if not ready:
        if pipeline_config.runtime.device_graph_capture:
            _logger.warning(
                "Device graph capture is on, but warmup_graph_capture() has "
                "not run on this pipeline, so every step runs eagerly. To "
                "replay captured graphs, call pipeline.warmup_graph_capture() "
                "first, or load the model with "
                "AutoModel(..., device_graph_capture=True)."
            )
        return
    if dp_degree > 1:
        raise _unsupported(
            pipeline_config,
            "device graph capture with data parallelism",
            "Replay needs every replica to step the same number of rows. "
            "Serve pads the replicas to keep them equal, and generate() does "
            "not pad",
            ["--device-graph-capture"],
            note=(
                "To run this here, load the model without device graph capture."
            ),
        )


def _check_penalties_enabled(
    pipeline_config: PipelineConfig, penalties: Mapping[str, float]
) -> None:
    """Raises if the caller set ``penalties`` the pipeline's sampler skips.

    ``penalties`` holds the penalty arguments the caller set to a non-default
    value.
    """
    if penalties and not pipeline_config.sampling.enable_penalties:
        requested = ", ".join(
            f"{name}={value}" for name, value in penalties.items()
        )
        raise ValueError(
            "This pipeline was built with enable_penalties=False, so its "
            f"sampler would ignore {requested}. Rebuild it with "
            "sampling=SamplingConfig(enable_penalties=True) (in max serve: "
            "--enable-penalties)."
        )


def _align_up(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def _check_batch_fits(
    pipeline: BatchPipeline,
    prompt_lengths: Sequence[int],
    dp_degree: int,
) -> None:
    """Raises if the batch needs more rows or tokens than the plan allows."""
    pipeline_config = pipeline.pipeline_config
    max_rows = pipeline.max_batch_size * dp_degree
    if len(prompt_lengths) > max_rows:
        raise _unsupported(
            pipeline_config,
            "scheduling across batches",
            f"{len(prompt_lengths)} prompts exceed the pipeline's max batch "
            f"size of {max_rows}"
            + (
                f" ({pipeline.max_batch_size} per replica x {dp_degree} "
                "replicas)"
                if dp_degree > 1
                else ""
            )
            + ", and generate() never splits its prompts",
            ["--max-batch-size", str(pipeline.max_batch_size)],
            note="To run this batch here, raise max_batch_size.",
        )
    model = pipeline._pipeline_model
    batch_processor = model.batch_processor
    # The scheduler chunks a prefill to fit this budget. Batch does not, and
    # an overflow would otherwise fail deep in graph input staging.
    token_budget = (
        None
        if batch_processor is None
        else batch_processor.runtime.max_batch_active_tokens // dp_degree
    )
    total_tokens_cap = model.planned_max_batch_total_tokens
    page_size = pipeline.kv_manager.params.page_size
    for replica_idx in range(dp_degree):
        lengths = prompt_lengths[replica_idx::dp_degree]
        if token_budget is not None and sum(lengths) > token_budget:
            raise _unsupported(
                pipeline_config,
                "chunked prefill",
                f"Replica {replica_idx}'s prompts total {sum(lengths)} "
                f"tokens, but the pipeline stages at most {token_budget} per "
                "replica in one forward",
                [
                    "--enable-chunked-prefill",
                    "--max-batch-input-tokens",
                    str(pipeline_config.runtime.max_batch_input_tokens),
                ],
                note="To run this batch here, raise max_batch_input_tokens.",
            )
        if total_tokens_cap is None or not lengths:
            continue
        # The MLA prefill plan starts each row at a page-aligned offset and
        # clamps the last row's padded tail at the buffer end, so only its
        # raw tokens must fit. This equals serve's TotalContextTokenBudget
        # when the cap is a page multiple.
        resident_tokens = (
            sum(_align_up(length, page_size) for length in lengths[:-1])
            + lengths[-1]
        )
        if resident_tokens > total_tokens_cap:
            raise _unsupported(
                pipeline_config,
                "scheduling across batches",
                f"Replica {replica_idx}'s prompts occupy {resident_tokens} "
                f"tokens with each row but the last padded to a "
                f"{page_size}-token page, but the plan caps a replica's "
                f"resident context at {total_tokens_cap} tokens",
                ["--max-batch-total-tokens", str(total_tokens_cap)],
                note="To run this batch here, raise max_batch_total_tokens.",
            )


class Batch:
    """Steps a fixed batch of prompts through a pipeline, one execute at a time.

    Batch runs a pipeline's forward passes over exactly the batch you pass,
    with nothing added on top of the pipeline API, which also makes it
    useful for reproducing and localizing kernel and model accuracy bugs.
    Serving features (scheduling and in-flight batching, chunked prefill,
    prefix caching, LoRA, and structured output) are tested through
    ``max serve``. Requesting one that Batch cannot run raises an error that
    gives the matching ``max serve`` command.

    Constructing the batch claims KV cache for every row. Use it as a
    context manager so an exception releases the rows. Row ``i`` samples
    with ``seed + i * 2**32``. Serve deliberately makes requests with the
    same pinned seed draw identically. Batch instead gives each row its own
    stream, as Hugging Face users expect, so identical prompts draw
    different samples, a rerun reproduces them, and row 0 matches serve for
    the same seed. A prompt's draw still depends on its row index.

    On a :attr:`synchronous` pipeline, each :meth:`step` returns its own
    forward's tokens and the device is idle between steps. A pipeline that
    is not synchronous returns the previous execute's tokens instead: the
    first step returns ``{}``, a row finishes one step after its last token
    is sampled, ``ignore_eos`` takes ``max_new_tokens + 1`` steps, and a
    forward can still be running between steps and after :attr:`done`. Such
    a pipeline allows one open batch at a time, and if its execute raises,
    reload it: closing the batch frees the rows but cannot repair the
    pipeline's pending state.

    ``ignore_eos=True`` is the batch mode closest to Hugging Face
    ``generate``, which keeps finished rows in the batch as padding. Prefill shapes still
    differ, because MAX batches prompts ragged where Hugging Face pads them
    on the left, so compare logits with a tolerance.

    With prefix caching on, a later batch can reuse KV an earlier one
    committed and prefill fewer tokens. Turn prefix caching off in the KV
    cache config when prefill shapes must repeat across runs.

    On a pipeline that ran ``warmup_graph_capture()``, the first (prefill)
    step runs eagerly and later steps replay captured graphs, as long as
    each replica has at most ``min(max_batch_size, 128)`` rows. Warm up
    before creating a Batch. With device graph capture on but no warmup,
    Batch logs a warning and runs every step eagerly. A logits processor
    receives the captured output buffer, which the next replay overwrites,
    so it must copy what it keeps.

    .. Skipped: the example needs a loaded pipeline.
    .. skip: next

    .. code-block:: python

        with Batch(pipeline, input_ids, max_new_tokens=20) as batch:
            for step_tokens in batch:
                print(step_tokens)
        sequences = batch.sequences

    Args:
        pipeline: The pipeline to run.
        input_ids: One token-id prompt per row, as lists or a 2D array.
        attention_mask: Zeros mark padding in ``input_ids``, which Batch
            drops, since it runs prompts unpadded. A tokenizer returns it
            with the ids.
        max_new_tokens: The most tokens to generate for each row.
        min_new_tokens: The fewest tokens a row generates before EOS can end
            it. Defaults to the model's generation config.
        temperature: The sampling temperature. Defaults to the pipeline's
            runtime default, then the model's generation config.
        top_k: The top-k limit, with the same fallback as ``temperature``.
        top_p: The nucleus sampling threshold. Defaults to the model's
            generation config.
        min_p: The minimum token probability relative to the most likely
            token. Defaults to 0 (off), since the generation config does
            not set it.
        repetition_penalty: The divisor for logits of repeated tokens.
            Defaults to the model's generation config.
        frequency_penalty: The penalty proportional to a token's count.
            Defaults to 0 (off), since the generation config does not set
            it.
        presence_penalty: The flat penalty for a token already present.
            Defaults to 0 (off), since the generation config does not set
            it.
        ignore_eos: Keeps generating past EOS, so every row runs
            ``max_new_tokens`` tokens and the batch size never changes.
        stop_token_ids: Token ids that end a row, in addition to EOS.
        seed: The sampling seed for row 0. Row ``i`` uses
            ``seed + i * 2**32``.

    Raises:
        TypeError: If ``input_ids`` or ``attention_mask`` is not a list of
            integer lists.
        ValueError: If the request needs a feature Batch leaves to
            ``max serve``: LoRA, device graph capture with ``data_parallel_degree > 1`` on
            a warmed-up pipeline, or a batch that only fits by scheduling or
            chunked prefill (more rows than the planned max batch size, or a
            replica's prompts over the staged token budget or the batch
            total-token cap). Also raised for a prompt with no
            room to generate within the planned max length, an invalid
            sampling value, penalties on a pipeline built without
            ``enable_penalties``, ``max_new_tokens`` below 1 or below
            ``min_new_tokens``, ``attention_mask`` with a different shape
            than ``input_ids``, ``ignore_eos`` with a row that does not fit
            or with speculative decoding, image inputs, or a model that does
            not generate text.
        RuntimeError: If the pipeline is not synchronous and another batch
            on it is still open.
    """

    def __init__(
        self,
        pipeline: BatchPipeline,
        input_ids: Iterable[Sequence[int]],
        attention_mask: Iterable[Sequence[int]] | None = None,
        *,
        max_new_tokens: int,
        min_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        ignore_eos: bool = False,
        stop_token_ids: Sequence[int] | None = None,
        seed: int = 0,
    ) -> None:
        prompts = _materialize_prompts(input_ids, attention_mask)
        pipeline_config = pipeline.pipeline_config
        penalties = {
            name: value
            for name, value, default in (
                ("repetition_penalty", repetition_penalty, 1.0),
                ("frequency_penalty", frequency_penalty, 0.0),
                ("presence_penalty", presence_penalty, 0.0),
            )
            if value is not None and value != default
        }
        _check_supported(pipeline_config)
        if ignore_eos and pipeline_config.speculative is not None:
            raise ValueError(
                "ignore_eos cannot keep the batch size fixed with speculative "
                "decoding, because a step can add several tokens to a row, so "
                "rows reach max_new_tokens at different steps."
            )
        if max_new_tokens < 1:
            raise ValueError(
                f"max_new_tokens must be at least 1, was {max_new_tokens}."
            )
        sampling_params = _resolve_sampling_params(
            pipeline,
            SamplingParamsInput(
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                ignore_eos=ignore_eos,
                stop_token_ids=(
                    None if stop_token_ids is None else list(stop_token_ids)
                ),
                seed=seed,
            ),
        )
        # After validation, so a bad penalty value reports itself.
        _check_penalties_enabled(pipeline_config, penalties)
        contexts = _build_contexts(
            pipeline,
            prompts,
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
        )
        # After the max length check: a too-long prompt also trips these
        # limits, and their advice cannot fix it.
        dp_degree = pipeline_config.model.data_parallel_degree
        _check_batch_fits(pipeline, [len(p) for p in prompts], dp_degree)
        _check_graph_capture(pipeline, dp_degree)
        synchronous = not (
            isinstance(pipeline, _OverlapSource) and pipeline.overlap_active
        )
        if not synchronous:
            open_batch_ref = _open_batches.get(pipeline)
            open_batch = (
                open_batch_ref() if open_batch_ref is not None else None
            )
            if open_batch is not None and not open_batch.done:
                raise RuntimeError(
                    "The pipeline is already running another batch, and a "
                    "pipeline that is not synchronous returns each step's "
                    "results to whichever batch steps next. Finish or close "
                    "that batch first."
                )
        self._pipeline = pipeline
        self._dp_degree = dp_degree
        self._synchronous = synchronous
        self._contexts = contexts
        self._prompts = prompts
        self._generated: list[list[int]] = [[] for _ in contexts]
        # Live rows in input order, mapped to their DP replica. Replicas
        # need no padding: Batch rejects data parallelism on a warmed-up
        # pipeline, and eager steps accept unequal replicas.
        self._live: dict[int, int] = {}
        try:
            for row, context in enumerate(contexts):
                replica_idx = row % dp_degree
                pipeline.kv_manager.claim(context, replica_idx=replica_idx)
                self._live[row] = replica_idx
        except BaseException:
            self.close()
            raise
        if not synchronous:
            _open_batches[pipeline] = weakref.ref(self)

    def __enter__(self) -> Batch:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __iter__(self) -> Iterator[dict[int, list[int]]]:
        """Steps until every row finishes, yielding each :meth:`step`'s tokens."""
        while not self.done:
            yield self.step()

    @property
    def synchronous(self) -> bool:
        """Whether each step returns its own forward's tokens, with no lag."""
        return self._synchronous

    @property
    def done(self) -> bool:
        """Whether every row has finished or been removed."""
        return not self._live

    @property
    def sequences(self) -> list[list[int]]:
        """Each row's prompt followed by its new tokens so far, in input order."""
        return [
            [*prompt, *generated]
            for prompt, generated in zip(
                self._prompts, self._generated, strict=True
            )
        ]

    def step(self) -> dict[int, list[int]]:
        """Runs exactly one ``pipeline.execute`` over the live rows.

        Rows the execute reports done are released before this returns. On a
        pipeline that is not synchronous the tokens lag one step, as
        :class:`Batch` describes.

        Returns:
            The tokens the execute reported for each live row, keyed by row
            index.

        Raises:
            RuntimeError: If no rows are live.
        """
        if self.done:
            raise RuntimeError("Batch has no live rows to step.")
        # Fresh lists every step: OverlapTextGenerationPipeline keeps a
        # reference to the previous step's batch, which must not change
        # underneath it.
        batches: list[list[TextContext]] = [[] for _ in range(self._dp_degree)]
        for row, replica_idx in self._live.items():
            context = self._contexts[row]
            # A KV connector's onload lands off the forward stream, so the
            # scheduler holds the row back until the onload lands. Batch
            # waits instead.
            self._pipeline.kv_manager.alloc(context).synchronize()
            batches[replica_idx].append(context)
        # Rebuilt each step so decode runs as TG, as under the scheduler.
        # Reusing one inputs object would leave every step CE.
        outputs = self._pipeline.execute(TextGenerationInputs(batches=batches))

        # Looking up live rows only drops the extra output a pipeline that
        # is not synchronous emits for a row already released.
        step_tokens: dict[int, list[int]] = {}
        for row in list(self._live):
            output = outputs.get(self._contexts[row].request_id)
            if output is None:
                continue
            tokens = [int(token) for token in output.tokens]
            self._generated[row].extend(tokens)
            step_tokens[row] = tokens
            if output.is_done:
                self._release(row)
        return step_tokens

    def remove(self, row: int) -> None:
        """Releases ``row`` so later steps run without it.

        Removing a row that already finished is a no-op.

        Args:
            row: The input index of the row to remove.

        Raises:
            IndexError: If ``row`` is not an input index.
        """
        if not 0 <= row < len(self._contexts):
            raise IndexError(
                f"Row {row} is out of range for {len(self._contexts)} rows."
            )
        if row in self._live:
            self._release(row)

    def close(self) -> None:
        """Releases every live row.

        Safe to call more than once. Every row is attempted even if one
        fails, and the first error is re-raised afterwards.
        """
        first_error: BaseException | None = None
        for row in list(self._live):
            try:
                self._release(row)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def _release(self, row: int) -> None:
        context = self._contexts[row]
        # Not a double free: the pipeline's release leaves the paged KV
        # cache alone and frees what the manager cannot, such as recurrent
        # state slots and encoder-cache refs.
        if self._pipeline.kv_manager.contains(context):
            self._pipeline.kv_manager.release(context)
        self._pipeline.release(context.request_id)
        # Dropped last, so a release that raises leaves the row to retry.
        del self._live[row]


def generate(
    pipeline: BatchPipeline,
    input_ids: Iterable[Sequence[int]],
    attention_mask: Iterable[Sequence[int]] | None = None,
    *,
    max_new_tokens: int,
    min_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    ignore_eos: bool = False,
    stop_token_ids: Sequence[int] | None = None,
    seed: int = 0,
) -> GenerateResult:
    """Generates tokens for a fixed batch of prompts, run as one batch.

    Steps a :class:`Batch` until every row finishes. The arguments and their
    defaults are the same as for :class:`Batch`, and the sampling arguments
    match the ``max generate`` flags of the same name.

    Args:
        pipeline: The pipeline to run.
        input_ids: One token-id prompt per row, as lists or a 2D array.
        attention_mask: Zeros mark padding in ``input_ids`` to drop.
        max_new_tokens: The most tokens to generate for each row.
        min_new_tokens: The fewest tokens a row generates before EOS can end
            it.
        temperature: The sampling temperature.
        top_k: The top-k limit.
        top_p: The nucleus sampling threshold.
        min_p: The minimum token probability relative to the most likely
            token.
        frequency_penalty: The penalty proportional to a token's count.
        presence_penalty: The flat penalty for a token already present.
        repetition_penalty: The divisor for logits of repeated tokens.
        ignore_eos: Keeps generating past EOS, so the batch size never
            changes.
        stop_token_ids: Token ids that end a row, in addition to EOS.
        seed: The sampling seed for row 0. Row ``i`` uses
            ``seed + i * 2**32``.

    Returns:
        Each row's prompt followed by its new tokens, in input order.

    Raises:
        TypeError: If ``input_ids`` or ``attention_mask`` is not a list of
            integer lists.
        ValueError: If the request needs a feature Batch leaves to
            ``max serve``, such as device graph capture with
            ``data_parallel_degree > 1`` on a warmed-up pipeline, or the
            batch does not fit the pipeline's plan, as described for
            :class:`Batch`.
        RuntimeError: If the pipeline is not synchronous and another batch
            on it is still open.
    """
    with Batch(
        pipeline,
        input_ids,
        attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        ignore_eos=ignore_eos,
        stop_token_ids=stop_token_ids,
        seed=seed,
    ) as batch:
        while not batch.done:
            batch.step()
        return GenerateResult(sequences=batch.sequences)
