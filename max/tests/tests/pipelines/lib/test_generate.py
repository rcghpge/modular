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
"""``Batch`` bookkeeping over fake pipelines.

The fakes stand in for a model because everything under test is bookkeeping
over ``execute``: which rows run each step, when a row's KV cache and
pipeline state are released, and whether a failed run leaves anything
claimed. Each prompt's first token is the step on which it emits EOS, and
the fakes update the real ``TextContext`` so EOS and max-length stops come
from the context itself, as they do in a real pipeline.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.driver import DeviceSpec
from max.pipelines.context import (
    GenerationStatus,
    PixelContext,
    SamplingParamsGenerationConfigDefaults,
    TextAndVisionContext,
    TextContext,
    TextGenerationOutput,
)
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.generate import Batch, generate
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.lib.registry import PIPELINE_REGISTRY
from max.pipelines.lora import LoRAConfig
from max.pipelines.modeling.types import RequestID, TextGenerationInputs
from max.pipelines.sampling import SamplingConfig, request_row_seed
from max.pipelines.speculative import SpeculativeConfig

EOS = 2
NEVER = 99
MODEL_PATH = "org/model"


@pytest.fixture(autouse=True)
def text_context_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolves the fake config to a text-only architecture."""
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_context_type",
        lambda pipeline_config: TextContext,
    )


@dataclasses.dataclass
class FakeTransfer:
    """An onload that lands only once synchronized."""

    landed: bool = False

    def is_complete(self) -> bool:
        return self.landed

    def synchronize(self) -> None:
        self.landed = True


@dataclasses.dataclass
class FakeKVParams:
    page_size: int = 4


@dataclasses.dataclass
class FakeKVManager:
    """Tracks claims by request id, since a context is not hashable."""

    params: FakeKVParams = dataclasses.field(default_factory=FakeKVParams)
    claimed: dict[RequestID, int] = dataclasses.field(default_factory=dict)
    contexts: list[TextContext] = dataclasses.field(default_factory=list)
    released: list[RequestID] = dataclasses.field(default_factory=list)
    transfers: dict[RequestID, FakeTransfer] = dataclasses.field(
        default_factory=dict
    )
    fail_on_claim: int | None = None

    def claim(self, ctx: TextContext, replica_idx: int = 0) -> None:
        if len(self.claimed) == self.fail_on_claim:
            raise RuntimeError("claim failed")
        self.claimed[ctx.request_id] = replica_idx
        self.contexts.append(ctx)

    def alloc(self, ctx: TextContext) -> FakeTransfer:
        assert ctx.request_id in self.claimed
        transfer = FakeTransfer()
        self.transfers[ctx.request_id] = transfer
        return transfer

    def release(self, ctx: TextContext) -> None:
        del self.claimed[ctx.request_id]
        self.released.append(ctx.request_id)

    def contains(self, ctx: TextContext) -> bool:
        return ctx.request_id in self.claimed


@dataclasses.dataclass
class FakeTokenizer:
    eos_token_ids: set[int] = dataclasses.field(default_factory=lambda: {EOS})
    tokenizer_vocab_size: int = 1000


@dataclasses.dataclass
class FakeRuntime:
    max_batch_active_tokens: int


@dataclasses.dataclass
class FakeBatchProcessor:
    runtime: FakeRuntime


@dataclasses.dataclass
class FakeModel:
    max_seq_len: int = 64
    batch_processor: FakeBatchProcessor | None = None
    planned_max_batch_total_tokens: int | None = None


class FakePipeline:
    """Emits EOS on the step a prompt's first token names, else step ids.

    Like ``TextGenerationPipeline``, it returns each step's tokens from the
    same ``execute``. It also checks that every row's KV onload landed
    before the forward reads it.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        data_parallel_degree: int = 1,
        max_seq_len: int = 64,
        max_batch_active_tokens: int | None = None,
        max_batch_total_tokens: int | None = None,
        page_size: int = 4,
        generation_config: SamplingParamsGenerationConfigDefaults | None = None,
        runtime_temperature: float | None = None,
        runtime_top_k: int | None = None,
        speculative: SpeculativeConfig | None = None,
        draft_model_path: str | None = None,
        lora: LoRAConfig | None = None,
        device_graph_capture: bool = False,
        enable_penalties: bool = False,
        tokenizer_vocab_size: int = 1000,
        device_specs: list[DeviceSpec] | None = None,
        quantization_encoding: str | None = None,
        max_length: int | None = None,
        max_length_user_provided: bool = False,
        trust_remote_code: bool = False,
        weight_path: list[Path] | None = None,
        weights_repo_id: str = MODEL_PATH,
    ) -> None:
        self.kv_manager = FakeKVManager(FakeKVParams(page_size))
        self.tokenizer = FakeTokenizer(
            tokenizer_vocab_size=tokenizer_vocab_size
        )
        self.max_batch_size = max_batch_size
        self._pipeline_model = FakeModel(
            max_seq_len,
            None
            if max_batch_active_tokens is None
            else FakeBatchProcessor(FakeRuntime(max_batch_active_tokens)),
            max_batch_total_tokens,
        )
        config = MagicMock(spec=PipelineConfig)
        config.model.model_path = MODEL_PATH
        config.model.data_parallel_degree = data_parallel_degree
        config.model.device_specs = device_specs or [DeviceSpec.accelerator(0)]
        config.model.quantization_encoding = quantization_encoding
        config.model.max_length = max_length
        config.model.max_length_is_user_provided = max_length_user_provided
        config.model.trust_remote_code = trust_remote_code
        config.model.weight_path = weight_path or []
        config.model.huggingface_weight_repo_id = weights_repo_id
        config.model.sampling_params_defaults = (
            generation_config or SamplingParamsGenerationConfigDefaults()
        )
        config.runtime = MagicMock(
            spec=PipelineRuntimeConfig,
            temperature=runtime_temperature,
            top_k=runtime_top_k,
            device_graph_capture=device_graph_capture,
            # A real runtime's max_batch_active_tokens sums the per-replica
            # max_batch_input_tokens over DP replicas.
            max_batch_input_tokens=None
            if max_batch_active_tokens is None
            else max_batch_active_tokens // data_parallel_degree,
        )
        config.speculative = speculative
        config.draft_model = (
            None
            if draft_model_path is None
            else MagicMock(model_path=draft_model_path)
        )
        config.lora = lora
        config.sampling = SamplingConfig(enable_penalties=enable_penalties)
        self.pipeline_config: PipelineConfig = config
        self.fail_on_step: int | None = None
        self.fail_next_release = False
        self.steps_run = 0
        # Rows per replica on each step.
        self.batch_sizes: list[list[int]] = []
        self.released: list[RequestID] = []

    def execute(
        self, inputs: TextGenerationInputs[TextContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        self.steps_run += 1
        if self.steps_run == self.fail_on_step:
            raise RuntimeError("execute interrupted")
        self.batch_sizes.append([len(batch) for batch in inputs.batches])
        for context in inputs.flat_batch:
            assert self.kv_manager.transfers[context.request_id].landed
        return self._forward(inputs.flat_batch)

    def _forward(
        self, contexts: list[TextContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        outputs: dict[RequestID, TextGenerationOutput] = {}
        for context in contexts:
            context.update(_next_token(context))
            outputs[context.request_id] = context.to_generation_output()
        return outputs

    def release(self, request_id: RequestID) -> None:
        if self.fail_next_release:
            self.fail_next_release = False
            raise RuntimeError("release failed")
        self.released.append(request_id)


class FakeOverlapDisabledPipeline(FakePipeline):
    """An overlap pipeline with overlap turned off, which runs synchronously."""

    overlap_active = False


class FakeOverlapPipeline(FakePipeline):
    """Returns the previous execute's tokens, like the overlap pipeline.

    Each forward appends a future token and is synced on the next execute,
    including rows released between executes.
    """

    overlap_active = True

    def __init__(self, device_graph_capture: bool = False) -> None:
        super().__init__(device_graph_capture=device_graph_capture)
        self._pending: list[tuple[TextContext, int]] = []

    def has_pending_outputs(self) -> bool:
        return bool(self._pending)

    def _forward(
        self, contexts: list[TextContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        # The new forward is sampled from the device before the previous
        # batch syncs, so it cannot see whether that batch ended a row.
        sampled = [(context, _next_token(context)) for context in contexts]
        outputs: dict[RequestID, TextGenerationOutput] = {}
        for context, token in self._pending:
            context.realize_future_token(token)
            outputs[context.request_id] = context.to_generation_output()
        for context in contexts:
            context.update_with_future_token()
        self._pending = sampled
        return outputs


class FakeWarmedUpPipeline(FakeOverlapDisabledPipeline):
    """A synchronous pipeline that ran ``warmup_graph_capture()``."""

    graph_capture_ready = True


class FakeWarmedUpOverlapPipeline(FakeOverlapPipeline):
    """A lagged pipeline that ran ``warmup_graph_capture()``."""

    graph_capture_ready = True


def _next_token(context: TextContext) -> int:
    step = context.tokens.generated_length + 1
    return EOS if step == int(context.tokens.prompt[0]) else 100 + step


def _prompts(*eos_steps: int) -> list[list[int]]:
    return [[eos_step, 7, 8] for eos_step in eos_steps]


def _new_tokens(sequences: list[list[int]]) -> list[list[int]]:
    """Drops the 3-token prompt :func:`_prompts` starts each row with."""
    return [sequence[3:] for sequence in sequences]


def _assert_points_to_serve(
    error: pytest.ExceptionInfo[ValueError], feature: str, flags: str = ""
) -> None:
    lines = str(error.value).splitlines()
    assert lines[0].startswith(f"generate() does not support {feature}. ")
    assert lines[1] == (
        "generate() runs the pipeline directly, outside max serve. "
        f"Test {feature} with this max serve command instead."
    )
    assert lines[2] == f"  max serve --model-path {MODEL_PATH}{flags}"


def _assert_nothing_held(pipeline: FakePipeline, num_rows: int) -> None:
    assert pipeline.kv_manager.claimed == {}, "a row was left claimed"
    assert len(pipeline.kv_manager.released) == num_rows
    assert len(set(pipeline.released)) == num_rows
    assert len(pipeline.released) == num_rows, "a row was released twice"


def test_finished_rows_leave_the_batch() -> None:
    pipeline = FakePipeline()
    with Batch(pipeline, _prompts(1, 3), max_new_tokens=5) as batch:
        assert batch.synchronous
        steps = []
        while not batch.done:
            steps.append(batch.step())

    assert steps == [{0: [EOS], 1: [101]}, {1: [102]}, {1: [EOS]}]
    assert pipeline.batch_sizes == [[2], [1], [1]]
    assert _new_tokens(batch.sequences) == [[EOS], [101, 102, EOS]]
    _assert_nothing_held(pipeline, 2)


def test_rows_stop_at_max_new_tokens() -> None:
    pipeline = FakePipeline()
    output = generate(pipeline, _prompts(NEVER, 2), max_new_tokens=3)

    assert _new_tokens(output.sequences) == [[101, 102, 103], [101, EOS]]
    assert pipeline.batch_sizes == [[2], [2], [1]]
    _assert_nothing_held(pipeline, 2)


def test_sequences_start_with_the_prompt() -> None:
    output = generate(FakePipeline(), _prompts(1, 3), max_new_tokens=5)
    assert output.sequences == [[1, 7, 8, EOS], [3, 7, 8, 101, 102, EOS]]


def test_iterating_a_batch_steps_it_to_the_end() -> None:
    pipeline = FakePipeline()
    with Batch(pipeline, _prompts(1, 3), max_new_tokens=5) as batch:
        steps = list(batch)

    assert steps == [{0: [EOS], 1: [101]}, {1: [102]}, {1: [EOS]}]
    assert batch.done


def test_attention_mask_drops_padding() -> None:
    pipeline = FakePipeline()
    output = generate(
        pipeline,
        [[0, 3, 7, 8], [1, 7, 8, 0]],
        attention_mask=[[0, 1, 1, 1], [1, 1, 1, 0]],
        max_new_tokens=5,
    )
    assert _new_tokens(output.sequences) == [[101, 102, EOS], [EOS]]


def test_attention_mask_of_another_shape_raises() -> None:
    pipeline = FakePipeline()
    with pytest.raises(ValueError, match="same shape as input_ids"):
        Batch(
            pipeline,
            _prompts(1, 3),
            attention_mask=[[1, 1, 1], [1, 1]],
            max_new_tokens=5,
        )
    assert pipeline.kv_manager.claimed == {}


def test_ignore_eos_keeps_every_row_to_max_new_tokens() -> None:
    pipeline = FakePipeline()
    with Batch(
        pipeline, _prompts(1, 3), max_new_tokens=5, ignore_eos=True
    ) as batch:
        for _ in batch:
            pass

    assert pipeline.batch_sizes == [[2]] * 5
    # As with serve's ignore_eos, tokens past EOS stay in the result.
    assert _new_tokens(batch.sequences) == [
        [EOS, 102, 103, 104, 105],
        [101, 102, EOS, 104, 105],
    ]
    _assert_nothing_held(pipeline, 2)


def test_ignore_eos_rejects_a_row_the_plan_would_clamp() -> None:
    pipeline = FakePipeline(max_seq_len=8)
    with pytest.raises(
        ValueError, match=r"planned max length is 8.*Raise max_length"
    ):
        Batch(pipeline, _prompts(1), max_new_tokens=6, ignore_eos=True)
    assert pipeline.kv_manager.claimed == {}


def test_stop_token_ids_end_a_row() -> None:
    output = generate(
        FakePipeline(), _prompts(NEVER), max_new_tokens=5, stop_token_ids=[102]
    )
    assert _new_tokens(output.sequences) == [[101, 102]]


def test_generation_config_min_new_tokens_over_max_new_tokens_raises() -> None:
    pipeline = FakePipeline(
        generation_config=SamplingParamsGenerationConfigDefaults(
            min_new_tokens=8
        )
    )
    with pytest.raises(
        ValueError,
        match=r"min_new_tokens=8, .* exceeds max_new_tokens=4.*Raise "
        r"max_new_tokens",
    ):
        Batch(pipeline, _prompts(NEVER), max_new_tokens=4)
    with pytest.raises(ValueError, match=r"min_new_tokens=6, "):
        Batch(
            FakePipeline(), _prompts(NEVER), max_new_tokens=4, min_new_tokens=6
        )
    assert pipeline.kv_manager.claimed == {}


def test_rows_round_robin_across_replicas() -> None:
    pipeline = FakePipeline(max_batch_size=2, data_parallel_degree=2)
    with Batch(pipeline, _prompts(1, 2, 3, 3), max_new_tokens=5) as batch:
        assert list(pipeline.kv_manager.claimed.values()) == [0, 1, 0, 1]
        while not batch.done:
            batch.step()

    assert pipeline.batch_sizes == [[2, 2], [1, 2], [1, 1]]
    _assert_nothing_held(pipeline, 4)


def test_remove_frees_the_row_and_the_rest_finish() -> None:
    pipeline = FakePipeline()
    with Batch(pipeline, _prompts(3, 3, 3), max_new_tokens=5) as batch:
        batch.step()
        batch.remove(1)
        assert len(pipeline.released) == 1
        assert pipeline.kv_manager.released == pipeline.released
        assert pipeline.released[0] not in pipeline.kv_manager.claimed
        batch.remove(1)
        while not batch.done:
            batch.step()

    assert pipeline.batch_sizes == [[3], [2], [2]]
    assert _new_tokens(batch.sequences) == [
        [101, 102, EOS],
        [101],
        [101, 102, EOS],
    ]
    _assert_nothing_held(pipeline, 3)


def test_remove_rejects_an_unknown_row() -> None:
    pipeline = FakePipeline()
    with Batch(pipeline, _prompts(1), max_new_tokens=5) as batch:
        with pytest.raises(IndexError):
            batch.remove(1)


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_interrupted_run_releases_rows_and_pipeline_is_reusable(
    error: type[BaseException],
) -> None:
    pipeline = FakePipeline()
    with pytest.raises(error):
        with Batch(pipeline, _prompts(4, 4), max_new_tokens=5) as batch:
            batch.step()
            raise error()
    _assert_nothing_held(pipeline, 2)

    output = generate(pipeline, _prompts(2, 2), max_new_tokens=5)

    assert _new_tokens(output.sequences) == [[101, EOS], [101, EOS]]
    _assert_nothing_held(pipeline, 4)


def test_execute_failure_releases_rows() -> None:
    pipeline = FakePipeline()
    pipeline.fail_on_step = 2
    with pytest.raises(RuntimeError, match="execute interrupted"):
        generate(pipeline, _prompts(4, 4), max_new_tokens=5)
    _assert_nothing_held(pipeline, 2)


def test_failed_claim_releases_rows_already_claimed() -> None:
    pipeline = FakePipeline()
    pipeline.kv_manager.fail_on_claim = 2
    with pytest.raises(RuntimeError, match="claim failed"):
        Batch(pipeline, _prompts(1, 1, 1), max_new_tokens=5)
    _assert_nothing_held(pipeline, 2)


def test_close_is_idempotent() -> None:
    pipeline = FakePipeline()
    batch = Batch(pipeline, _prompts(4, 4), max_new_tokens=5)
    batch.close()
    batch.close()

    assert batch.done
    _assert_nothing_held(pipeline, 2)
    with pytest.raises(RuntimeError, match="no live rows"):
        batch.step()


def test_close_releases_every_row_when_one_release_fails() -> None:
    pipeline = FakePipeline()
    batch = Batch(pipeline, _prompts(4, 4, 4), max_new_tokens=5)
    pipeline.fail_next_release = True
    with pytest.raises(RuntimeError, match="release failed"):
        batch.close()

    assert pipeline.kv_manager.claimed == {}
    assert len(pipeline.released) == 2
    assert not batch.done
    batch.close()
    assert batch.done
    _assert_nothing_held(pipeline, 3)


def test_out_of_vocab_token_raises() -> None:
    pipeline = FakePipeline(tokenizer_vocab_size=102)
    with pytest.raises(RuntimeError, match="out-of-vocabulary token_id=102"):
        generate(pipeline, _prompts(NEVER), max_new_tokens=3)
    _assert_nothing_held(pipeline, 1)


def test_max_new_tokens_below_one_raises() -> None:
    pipeline = FakePipeline()
    with pytest.raises(ValueError, match="at least 1, was 0"):
        Batch(pipeline, _prompts(1), max_new_tokens=0)
    assert pipeline.kv_manager.claimed == {}


@pytest.mark.parametrize(
    ("max_batch_size", "data_parallel_degree", "num_rows", "max_rows"),
    [
        (2, 1, 3, "2"),
        (2, 2, 5, "4 (2 per replica x 2 replicas)"),
    ],
)
def test_oversize_batch_raises_before_claiming(
    max_batch_size: int,
    data_parallel_degree: int,
    num_rows: int,
    max_rows: str,
) -> None:
    pipeline = FakePipeline(
        max_batch_size=max_batch_size, data_parallel_degree=data_parallel_degree
    )
    with pytest.raises(ValueError, match="raise max_batch_size") as error:
        Batch(pipeline, _prompts(*[1] * num_rows), max_new_tokens=5)
    assert str(error.value).splitlines()[0] == (
        "generate() does not support scheduling across batches. "
        f"{num_rows} prompts exceed the pipeline's max batch size of "
        f"{max_rows}, and generate() never splits its prompts."
    )
    dp_flag = (
        ""
        if data_parallel_degree == 1
        else f" --data-parallel-degree {data_parallel_degree}"
    )
    _assert_points_to_serve(
        error,
        "scheduling across batches",
        f"{dp_flag} --max-batch-size {max_batch_size}",
    )
    assert pipeline.kv_manager.claimed == {}


def test_prefill_over_a_replica_token_budget_raises() -> None:
    # 12 staged tokens over 2 replicas leaves 6 per replica: two prompts.
    pipeline = FakePipeline(data_parallel_degree=2, max_batch_active_tokens=12)
    generate(pipeline, _prompts(1, 1, 1, 1), max_new_tokens=5)

    with pytest.raises(
        ValueError,
        match=r"(?s)chunked prefill\. Replica 0's prompts total 9 tokens"
        r".*raise max_batch_input_tokens",
    ) as error:
        Batch(pipeline, _prompts(1, 1, 1, 1, 1), max_new_tokens=5)
    _assert_points_to_serve(
        error,
        "chunked prefill",
        " --data-parallel-degree 2 --enable-chunked-prefill"
        " --max-batch-input-tokens 6",
    )
    _assert_nothing_held(pipeline, 4)


@pytest.mark.parametrize(
    ("context_type", "message"),
    [
        (TextAndVisionContext, "does not support image inputs yet"),
        (PixelContext, "runs text generation models only"),
    ],
)
def test_models_that_take_or_make_images_are_not_supported_yet(
    monkeypatch: pytest.MonkeyPatch, context_type: type, message: str
) -> None:
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_context_type",
        lambda pipeline_config: context_type,
    )
    pipeline = FakePipeline()
    with pytest.raises(ValueError, match=message) as error:
        Batch(pipeline, _prompts(1), max_new_tokens=5)
    assert context_type.__name__ in str(error.value)
    assert "max serve" not in str(error.value)
    assert pipeline.kv_manager.claimed == {}


def test_overlap_results_lag_one_step() -> None:
    pipeline = FakeOverlapPipeline()
    with Batch(pipeline, _prompts(1, 3), max_new_tokens=5) as batch:
        assert not batch.synchronous
        steps = []
        while not batch.done:
            steps.append(batch.step())

    assert steps == [{}, {0: [EOS], 1: [101]}, {1: [102]}, {1: [EOS]}]
    # Each row runs one more forward after the one that ended it.
    assert pipeline.batch_sizes == [[2], [2], [1], [1]]
    assert _new_tokens(batch.sequences) == [[EOS], [101, 102, EOS]]
    assert pipeline.has_pending_outputs()
    _assert_nothing_held(pipeline, 2)


def test_overlap_ignore_eos_takes_one_extra_step() -> None:
    pipeline = FakeOverlapPipeline()
    output = generate(
        pipeline, _prompts(1, 3), max_new_tokens=3, ignore_eos=True
    )

    assert pipeline.batch_sizes == [[2]] * 4
    assert _new_tokens(output.sequences) == [[EOS, 102, 103], [101, 102, EOS]]
    _assert_nothing_held(pipeline, 2)


def test_overlap_pending_batch_after_close_leaves_next_batch_clean() -> None:
    pipeline = FakeOverlapPipeline()
    with Batch(pipeline, _prompts(4, 4), max_new_tokens=5) as batch:
        batch.step()
        batch.step()
    assert pipeline.has_pending_outputs()
    _assert_nothing_held(pipeline, 2)

    output = generate(pipeline, _prompts(2, 2), max_new_tokens=5)

    assert _new_tokens(output.sequences) == [[101, EOS], [101, EOS]]
    _assert_nothing_held(pipeline, 4)


def _prompt_of_length(length: int) -> list[int]:
    return [NEVER] + [7] * (length - 1)


def test_batch_total_tokens_cap_charges_page_aligned_rows() -> None:
    # Pages of 4: two 3-token rows cost 4 + 3, a third pushes past 10.
    pipeline = FakePipeline(max_batch_total_tokens=10, page_size=4)
    generate(pipeline, _prompts(1, 1), max_new_tokens=5)

    with pytest.raises(
        ValueError,
        match=r"(?s)11 tokens.*caps.*10.*raise max_batch_total_tokens",
    ) as error:
        Batch(pipeline, _prompts(1, 1, 1), max_new_tokens=5)
    _assert_points_to_serve(
        error, "scheduling across batches", " --max-batch-total-tokens 10"
    )
    _assert_nothing_held(pipeline, 2)


def test_batch_total_tokens_cap_is_per_replica() -> None:
    pipeline = FakePipeline(
        data_parallel_degree=2, max_batch_total_tokens=10, page_size=4
    )
    generate(pipeline, _prompts(1, 1, 1, 1), max_new_tokens=5)

    with pytest.raises(
        ValueError, match="scheduling across batches\\. Replica 0's prompts"
    ):
        Batch(pipeline, _prompts(1, 1, 1, 1, 1), max_new_tokens=5)


def test_batch_total_tokens_cap_skips_a_replica_with_no_rows() -> None:
    pipeline = FakePipeline(
        data_parallel_degree=2, max_batch_total_tokens=10, page_size=4
    )
    output = generate(pipeline, _prompts(2), max_new_tokens=5)
    assert _new_tokens(output.sequences) == [[101, EOS]]


def test_batch_total_tokens_cap_admits_a_lone_row_under_the_cap() -> None:
    # Page-aligned, a 9-token row would cost 12, but nothing sits after it.
    pipeline = FakePipeline(max_batch_total_tokens=10, page_size=4)
    prompt = _prompt_of_length(9)
    output = generate(pipeline, [prompt], max_new_tokens=1)
    assert output.sequences == [[*prompt, 101]]

    with pytest.raises(ValueError, match="11 tokens"):
        Batch(pipeline, [_prompt_of_length(11)], max_new_tokens=1)


def test_speculative_pipeline_runs_but_rejects_ignore_eos() -> None:
    pipeline = FakePipeline(
        speculative=SpeculativeConfig(
            speculative_method="eagle", num_speculative_tokens=3
        ),
        draft_model_path="org/draft",
    )
    output = generate(pipeline, _prompts(2), max_new_tokens=5)
    assert _new_tokens(output.sequences) == [[101, EOS]]

    with pytest.raises(ValueError, match="ignore_eos cannot keep"):
        Batch(pipeline, _prompts(1), max_new_tokens=5, ignore_eos=True)
    _assert_nothing_held(pipeline, 1)


def test_lora_pipeline_points_to_serve() -> None:
    pipeline = FakePipeline(
        lora=LoRAConfig(
            enable_lora=True,
            lora_paths=["sql=/adapters/sql", "/b"],
            max_lora_rank=64,
            max_num_loras=2,
        )
    )
    with pytest.raises(ValueError) as error:
        Batch(pipeline, _prompts(1), max_new_tokens=5)
    _assert_points_to_serve(
        error,
        "LoRA adapters",
        " --enable-lora --no-enable-prefix-caching --lora-paths"
        " sql=/adapters/sql --lora-paths /b --max-lora-rank 64"
        " --max-num-loras 2",
    )
    assert str(error.value).splitlines()[3] == (
        'Then send requests with "model": "<adapter name>". An adapter\'s '
        'name is the part of its --lora-paths entry before "=", or the '
        'whole entry if it has no "=" (names here: "sql", "/b").'
    )
    assert pipeline.kv_manager.claimed == {}


def test_lora_pipeline_without_static_adapters_points_to_load_endpoint() -> (
    None
):
    pipeline = FakePipeline(lora=LoRAConfig(enable_lora=True, lora_paths=[]))
    with pytest.raises(ValueError) as error:
        Batch(pipeline, _prompts(1), max_new_tokens=5)
    _assert_points_to_serve(
        error,
        "LoRA adapters",
        " --enable-lora --no-enable-prefix-caching --max-lora-rank 16"
        " --max-num-loras 1",
    )
    assert str(error.value).splitlines()[3] == (
        "Load an adapter with POST /v1/load_lora_adapter, then send "
        'requests with "model": "<lora_name>".'
    )
    assert pipeline.kv_manager.claimed == {}


def _graph_capture_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "max.pipelines"
        and record.levelno == logging.WARNING
        and "warmup_graph_capture()" in record.getMessage()
    ]


def test_graph_capture_without_warmup_warns_and_runs_eagerly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    expected = generate(FakePipeline(), _prompts(1, 3), max_new_tokens=5)
    assert _graph_capture_warnings(caplog) == []

    pipeline = FakePipeline(device_graph_capture=True)
    output = generate(pipeline, _prompts(1, 3), max_new_tokens=5)

    assert output.sequences == expected.sequences
    assert _graph_capture_warnings(caplog) == [
        "Device graph capture is on, but warmup_graph_capture() has not "
        "run on this pipeline, so every step runs eagerly. To replay "
        "captured graphs, call pipeline.warmup_graph_capture() first, or "
        "load the model with AutoModel(..., device_graph_capture=True)."
    ]
    _assert_nothing_held(pipeline, 2)


@pytest.mark.parametrize(
    "pipeline_type", [FakeWarmedUpPipeline, FakeWarmedUpOverlapPipeline]
)
def test_warmed_up_pipeline_runs_without_warning(
    pipeline_type: type[FakeWarmedUpPipeline | FakeWarmedUpOverlapPipeline],
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline = pipeline_type(device_graph_capture=True)
    output = generate(pipeline, _prompts(1, 3), max_new_tokens=5)

    assert _new_tokens(output.sequences) == [[EOS], [101, 102, EOS]]
    assert _graph_capture_warnings(caplog) == []
    _assert_nothing_held(pipeline, 2)


def test_warmed_up_pipeline_with_data_parallelism_points_to_serve(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline = FakeWarmedUpPipeline(
        data_parallel_degree=2, device_graph_capture=True
    )
    with pytest.raises(ValueError) as error:
        Batch(pipeline, _prompts(1, 1), max_new_tokens=5)
    _assert_points_to_serve(
        error,
        "device graph capture with data parallelism",
        " --data-parallel-degree 2 --device-graph-capture",
    )
    assert str(error.value).splitlines()[3] == (
        "To run this here, load the model without device graph capture."
    )
    assert pipeline.kv_manager.contexts == []
    assert pipeline.kv_manager.claimed == {}
    assert _graph_capture_warnings(caplog) == []


def test_data_parallel_without_warmup_runs_eagerly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline = FakePipeline(
        max_batch_size=2, data_parallel_degree=2, device_graph_capture=True
    )
    with Batch(pipeline, _prompts(1, 2, 3, 3), max_new_tokens=5) as batch:
        assert list(pipeline.kv_manager.claimed.values()) == [0, 1, 0, 1]
        while not batch.done:
            batch.step()

    assert pipeline.batch_sizes == [[2, 2], [1, 2], [1, 1]]
    assert len(_graph_capture_warnings(caplog)) == 1
    _assert_nothing_held(pipeline, 4)


def test_warmed_up_with_capture_off_in_config_still_checks_dp(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline = FakeWarmedUpPipeline(data_parallel_degree=2)
    with pytest.raises(
        ValueError, match="device graph capture with data parallelism"
    ):
        Batch(pipeline, _prompts(1, 1), max_new_tokens=5)
    assert pipeline.kv_manager.contexts == []

    output = generate(FakeWarmedUpPipeline(), _prompts(2), max_new_tokens=5)
    assert _new_tokens(output.sequences) == [[101, EOS]]
    assert _graph_capture_warnings(caplog) == []


@pytest.mark.parametrize(
    ("make_pipeline", "flags"),
    [
        (
            lambda: FakePipeline(
                max_batch_size=1, device_specs=[DeviceSpec.cpu()]
            ),
            " --devices cpu",
        ),
        (
            lambda: FakePipeline(
                max_batch_size=1, device_specs=[DeviceSpec.accelerator(1)]
            ),
            " --devices gpu:1",
        ),
        (
            lambda: FakePipeline(
                max_batch_size=1,
                device_specs=[
                    DeviceSpec.accelerator(0),
                    DeviceSpec.accelerator(1),
                ],
                data_parallel_degree=2,
                quantization_encoding="bfloat16",
                max_length=4096,
                max_length_user_provided=True,
            ),
            " --devices gpu:0,1 --quantization-encoding bfloat16"
            " --data-parallel-degree 2 --max-length 4096",
        ),
        # The architecture resolved this length, so serve resolves it too.
        (lambda: FakePipeline(max_batch_size=1, max_length=2048), ""),
        (
            lambda: FakePipeline(max_batch_size=1, trust_remote_code=True),
            " --trust-remote-code",
        ),
        # A bare weights repo id leaves only discovered files, so the
        # command names the repo.
        (
            lambda: FakePipeline(
                max_batch_size=1,
                weight_path=[Path("model.safetensors")],
                weights_repo_id="org/weights",
            ),
            " --weight-path org/weights",
        ),
        # Serve discovers the same default weight files.
        (
            lambda: FakePipeline(
                max_batch_size=1, weight_path=[Path("model.safetensors")]
            ),
            "",
        ),
    ],
)
def test_serve_command_repeats_the_pipeline_settings(
    make_pipeline: Callable[[], FakePipeline], flags: str
) -> None:
    pipeline = make_pipeline()
    with pytest.raises(ValueError) as error:
        Batch(pipeline, _prompts(1, 1, 1), max_new_tokens=5)
    _assert_points_to_serve(
        error, "scheduling across batches", f"{flags} --max-batch-size 1"
    )


def test_serve_command_puts_feature_flags_after_pipeline_settings() -> None:
    pipeline = FakePipeline(
        device_specs=[DeviceSpec.accelerator(0), DeviceSpec.accelerator(1)],
        data_parallel_degree=2,
        quantization_encoding="float8_e4m3fn",
        max_length=8192,
        max_length_user_provided=True,
        max_batch_size=1,
    )
    with pytest.raises(ValueError) as error:
        Batch(pipeline, _prompts(1, 1, 1), max_new_tokens=5)
    assert str(error.value).splitlines()[2] == (
        "  max serve --model-path org/model --devices gpu:0,1"
        " --quantization-encoding float8_e4m3fn --data-parallel-degree 2"
        " --max-length 8192 --max-batch-size 1"
    )


@pytest.mark.parametrize(
    ("repetition", "frequency", "presence", "requested"),
    [
        (1.1, None, None, "repetition_penalty=1.1"),
        (None, 0.2, None, "frequency_penalty=0.2"),
        (None, None, 0.3, "presence_penalty=0.3"),
    ],
)
def test_penalties_without_enable_penalties_raise(
    repetition: float | None,
    frequency: float | None,
    presence: float | None,
    requested: str,
) -> None:
    pipeline = FakePipeline()
    with pytest.raises(
        ValueError,
        match=rf"enable_penalties=False, so its sampler would ignore "
        rf"{requested}\..*sampling=SamplingConfig\(enable_penalties=True\)"
        r".*--enable-penalties",
    ) as error:
        Batch(
            pipeline,
            _prompts(1),
            max_new_tokens=5,
            repetition_penalty=repetition,
            frequency_penalty=frequency,
            presence_penalty=presence,
        )
    # Batch runs penalties on a pipeline that enables them.
    assert "does not support" not in str(error.value)
    assert pipeline.kv_manager.claimed == {}


@pytest.mark.parametrize(
    ("repetition", "frequency", "message"),
    [
        (0.0, None, "repetition_penalty must be a finite"),
        (None, 5.0, r"frequency_penalty must be in \[-2.0"),
    ],
)
def test_invalid_penalty_reports_the_value_before_enable_penalties(
    repetition: float | None, frequency: float | None, message: str
) -> None:
    pipeline = FakePipeline()
    with pytest.raises(ValueError, match=message):
        Batch(
            pipeline,
            _prompts(1),
            max_new_tokens=5,
            repetition_penalty=repetition,
            frequency_penalty=frequency,
        )
    assert pipeline.kv_manager.claimed == {}


def test_default_penalties_run_without_enable_penalties() -> None:
    pipeline = FakePipeline()
    output = generate(
        pipeline,
        _prompts(2),
        max_new_tokens=5,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    assert _new_tokens(output.sequences) == [[101, EOS]]


@pytest.mark.parametrize(
    "input_ids", [[1, 7, 8], np.array([1, 7, 8]), [[1, 7], "ab"]]
)
def test_input_ids_that_are_not_token_lists_raise_type_error(
    input_ids: list[list[int]],
) -> None:
    pipeline = FakePipeline()
    with pytest.raises(TypeError, match=r"Wrap one prompt as \[ids\]"):
        Batch(pipeline, input_ids, max_new_tokens=5)
    assert pipeline.kv_manager.claimed == {}


def test_input_ids_accepts_a_generator() -> None:
    pipeline = FakePipeline()
    output = generate(
        pipeline, (prompt for prompt in _prompts(1, 3)), max_new_tokens=5
    )
    assert _new_tokens(output.sequences) == [[EOS], [101, 102, EOS]]


def test_prompt_at_planned_max_length_raises() -> None:
    pipeline = FakePipeline(max_seq_len=3)
    with pytest.raises(
        ValueError,
        match=r"Row 0 needs at least 4 tokens.*planned max length.*"
        r"Raise max_length",
    ):
        Batch(pipeline, _prompts(NEVER), max_new_tokens=5)
    assert pipeline.kv_manager.claimed == {}


@pytest.mark.parametrize(
    ("max_batch_active_tokens", "max_batch_total_tokens"),
    [(8, None), (None, 8)],
)
def test_prompt_over_planned_max_length_raises_before_batch_limits(
    max_batch_active_tokens: int | None, max_batch_total_tokens: int | None
) -> None:
    # The prompt also exceeds the staged-token budget or total-token cap,
    # but only raising max_length would let it run.
    pipeline = FakePipeline(
        max_seq_len=8,
        max_batch_active_tokens=max_batch_active_tokens,
        max_batch_total_tokens=max_batch_total_tokens,
    )
    with pytest.raises(ValueError, match="Raise max_length") as error:
        Batch(pipeline, [_prompt_of_length(9)], max_new_tokens=1)
    assert "max serve" not in str(error.value)
    assert pipeline.kv_manager.claimed == {}


def test_prompt_one_under_planned_max_length_generates_one_token() -> None:
    pipeline = FakePipeline(max_seq_len=4)
    output = generate(pipeline, _prompts(NEVER), max_new_tokens=5)
    assert _new_tokens(output.sequences) == [[101]]


def test_generation_config_min_new_tokens_counts_against_max_length() -> None:
    # 3 prompt + 1 new fits in 5, but 3 prompt + 3 required new does not.
    pipeline = FakePipeline(
        max_seq_len=5,
        generation_config=SamplingParamsGenerationConfigDefaults(
            min_new_tokens=3
        ),
    )
    with pytest.raises(
        ValueError, match=r"needs at least 6 tokens.*Raise max_length"
    ):
        Batch(pipeline, _prompts(NEVER), max_new_tokens=5)
    assert pipeline.kv_manager.claimed == {}


def test_rows_are_clamped_to_the_planned_max_length() -> None:
    pipeline = FakePipeline(max_seq_len=8)
    output = generate(pipeline, _prompts(NEVER), max_new_tokens=10)

    assert _new_tokens(output.sequences) == [[101, 102, 103, 104, 105]]
    (context,) = pipeline.kv_manager.contexts
    assert context.status == GenerationStatus.MAXIMUM_LENGTH


@pytest.mark.parametrize("entry", [Batch, generate])
def test_sampling_knobs_reach_every_context(
    entry: Callable[..., object],
) -> None:
    pipeline = FakePipeline(enable_penalties=True)
    result = entry(
        pipeline,
        _prompts(1, 1),
        max_new_tokens=5,
        temperature=0.5,
        top_k=20,
        top_p=0.9,
        min_p=0.05,
        repetition_penalty=1.1,
        frequency_penalty=0.2,
        presence_penalty=0.3,
        min_new_tokens=2,
        stop_token_ids=[42],
    )
    if isinstance(result, Batch):
        result.close()

    assert len(pipeline.kv_manager.contexts) == 2
    for context in pipeline.kv_manager.contexts:
        params = context.sampling_params
        assert params.temperature == 0.5
        assert params.top_k == 20
        assert params.top_p == 0.9
        assert params.min_p == 0.05
        assert params.repetition_penalty == 1.1
        assert params.frequency_penalty == 0.2
        assert params.presence_penalty == 0.3
        assert params.min_new_tokens == 2
        assert params.stop_token_ids == [42]


def test_unset_sampling_knobs_fall_back_like_serve() -> None:
    pipeline = FakePipeline(
        generation_config=SamplingParamsGenerationConfigDefaults(
            temperature=0.7, top_k=50, top_p=0.8, repetition_penalty=1.2
        ),
        runtime_temperature=0.0,
    )
    with Batch(pipeline, _prompts(1), max_new_tokens=5):
        pass
    with Batch(pipeline, _prompts(1), max_new_tokens=5, temperature=0.3):
        pass

    runtime_default, caller_override = (
        context.sampling_params for context in pipeline.kv_manager.contexts
    )
    assert runtime_default.temperature == 0.0
    assert caller_override.temperature == 0.3
    assert caller_override.top_k == 50
    assert caller_override.top_p == 0.8
    assert caller_override.repetition_penalty == 1.2


def test_unset_top_k_falls_back_to_the_runtime_default() -> None:
    pipeline = FakePipeline(
        generation_config=SamplingParamsGenerationConfigDefaults(
            temperature=0.7, top_k=50
        ),
        runtime_top_k=5,
    )
    with Batch(pipeline, _prompts(1), max_new_tokens=5):
        pass
    with Batch(pipeline, _prompts(1), max_new_tokens=5, top_k=20):
        pass

    runtime_default, caller_override = (
        context.sampling_params for context in pipeline.kv_manager.contexts
    )
    assert runtime_default.temperature == 0.7
    assert runtime_default.top_k == 5
    assert caller_override.top_k == 20


def test_each_row_samples_with_its_own_seed() -> None:
    pipeline = FakePipeline()
    with Batch(pipeline, _prompts(1, 1, 1), max_new_tokens=5, seed=5):
        pass

    seeds = [
        context.sampling_params.seed for context in pipeline.kv_manager.contexts
    ]
    assert seeds == [5, 5 + 2**32, 5 + 2 * 2**32]


def test_rows_offset_by_their_length_draw_distinct_keys() -> None:
    # 4 + 0 == 3 + 1, so seeds one apart would give both rows one key.
    pipeline = FakePipeline()
    with Batch(
        pipeline, [_prompt_of_length(4), _prompt_of_length(3)], max_new_tokens=5
    ):
        pass

    first, second = pipeline.kv_manager.contexts
    assert request_row_seed(first) != request_row_seed(second)


def test_overlap_pipeline_allows_one_open_batch() -> None:
    pipeline = FakeOverlapPipeline()
    first = Batch(pipeline, _prompts(2), max_new_tokens=5)
    with pytest.raises(RuntimeError, match="already running another batch"):
        Batch(pipeline, _prompts(2), max_new_tokens=5)
    assert len(pipeline.kv_manager.claimed) == 1

    first.close()
    output = generate(pipeline, _prompts(2), max_new_tokens=5)
    assert _new_tokens(output.sequences) == [[101, EOS]]


@pytest.mark.parametrize(
    "pipeline_type", [FakePipeline, FakeOverlapDisabledPipeline]
)
def test_synchronous_pipeline_allows_concurrent_batches(
    pipeline_type: type[FakePipeline],
) -> None:
    pipeline = pipeline_type()
    with Batch(pipeline, _prompts(2), max_new_tokens=5) as first:
        with Batch(pipeline, _prompts(3), max_new_tokens=5) as second:
            assert first.synchronous
            assert second.synchronous
            assert first.step() == {0: [101]}
            while not (first.done and second.done):
                for batch in (first, second):
                    if not batch.done:
                        batch.step()

    assert _new_tokens(first.sequences) == [[101, EOS]]
    assert _new_tokens(second.sequences) == [[101, 102, EOS]]
