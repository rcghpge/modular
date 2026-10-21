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
"""Scores MAX's ModuleV3 MiMo-V2 port's logits on the tiny fixture against HF
goldens.

The ModuleV3 counterpart of ``../mimo_v2/test_serving_numerics_gpu.py``. It
runs the same checkpoint and the same goldens (the fixture is arch-agnostic:
what differs is only the compiled graph the logits come from), through the
V3 runner: ``Module.compile`` over a ``DeviceMesh``, no signal buffers, no
per-layer hidden states, no fused-QKV variants to sweep (the fused chunked
QKV path is not ported), no device-graph capture (the arch declares
``supports_device_graph_capture`` false and the graph has no capture
timeline). Both prompts run as one ragged batch, prefilled in chunks through
the paged KV cache and then decoded a token at a time.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import tiny_fixture
from max.driver import accelerator_count
from mimo_v3_runner import Runner

GOLDENS = (
    Path(__file__).parent.parent / "mimo_v2" / "testdata" / "tiny_goldens.npz"
)
"""The V2 harness's goldens: HF float32 logits of the same tiny checkpoint,
computed by ``make_random_fixture.py --tiny-goldens``. Reused, not copied —
the fixture checkpoint and goldens are arch-agnostic."""

CHUNK, DECODE = 96, 4

MEAN_KL, MAX_KL, TOP1 = 4e-3, 2.5e-2, 0.92
"""Bounds on KL(HF || MAX) over the whole vocabulary, and on top-1
agreement, over every row of both prompts.

The same bounds as the V2 test (see its docstring for the measured TP2
actuals: mean KL 1.09e-3, maximum KL 6.6e-3, top-1 0.972, and the defects
each bound catches by 6x or more). The goldens decode the same quantized
weights in float32, so the only difference from the V2 arm is the compiled
ModuleV3 graph: its residual reduces through ``Tensor.to(mapping)`` fused
with the following RMSNorm (``multiply_before_cast``, where the V2 graph
scales in BF16 after the residual add) and its MoE layers share one compiled
subgraph. The measured TP2 B200 actuals for this arm: mean KL 1.09e-3,
maximum KL 6.80e-3, top-1 0.972 — every bound clears by 3x or more, so a
touched graph shows up here before it moves the V2-vs-V3 parity gate.

The porting bugs the no-GPU lane caught (missing ``o_proj``, the field-only
E8M0 retyping) do not survive the runner at all, so this gate guards the
graph's numerics, not the load contract; the CPU harness test pins the
latter, and ``test_v2_v3_parity_gpu.py`` compares against the graph-API
arch directly."""


def _kl(
    gold: npt.NDArray[np.float32], test: npt.NDArray[np.float32]
) -> npt.NDArray[np.float64]:
    """Each row's KL(gold || test)."""
    lg, lt = (
        x.astype(np.float64)
        - np.logaddexp.reduce(x.astype(np.float64), -1, keepdims=True)
        for x in (gold, test)
    )
    return (np.exp(lg) * (lg - lt)).sum(-1)


@pytest.fixture(scope="module")
def logits(
    tmp_path_factory: pytest.TempPathFactory,
) -> list[npt.NDArray[np.float32]]:
    """The V3 model's logits for both prompts, teacher-forced at TP2."""
    checkpoint = tiny_fixture.tensors()
    goldens = np.load(GOLDENS)
    assert str(goldens["digest"]) == tiny_fixture.digest(checkpoint), (
        "tiny_fixture.py no longer builds the checkpoint the goldens were "
        "made from; rerun make_random_fixture.py --tiny-goldens."
    )
    path = tmp_path_factory.mktemp("mimo_v3")
    tiny_fixture.write(path, checkpoint)
    num_devices = int(
        os.environ.get("MIMO_TEST_TP_DEGREE", str(min(2, accelerator_count())))
    )
    assert accelerator_count() >= num_devices
    model = Runner(
        path,
        num_devices=num_devices,
        max_seq_len=256,
        kv_cache_bytes=256 * 1024**2,
    )
    prompts = [
        np.array(ids, np.int64) for ids in tiny_fixture.prompts().values()
    ]
    outputs = model.teacher_forced(prompts, CHUNK, DECODE)
    return [output for output in outputs if output is not None]


def test_serving_logits_match_hf(
    logits: list[npt.NDArray[np.float32]],
) -> None:
    goldens = np.load(GOLDENS)
    num_devices = int(
        os.environ.get("MIMO_TEST_TP_DEGREE", str(min(2, accelerator_count())))
    )
    prompts = tiny_fixture.prompts()
    kl, top1 = {}, {}
    for name, output in zip(prompts, logits, strict=True):
        gold = goldens[f"logits_{name}"]
        kl[name] = _kl(gold, output)
        top1[name] = gold.argmax(-1) == output.argmax(-1)
    rows = np.concatenate(list(kl.values()))
    agree = np.concatenate(list(top1.values())).mean()
    edge = tiny_fixture.BEACON_ROW + tiny_fixture.SLIDING_WINDOW
    summary = (
        f"{num_devices} GPU(s): KL mean {rows.mean():.2e}, max "
        f"{rows.max():.2e} (row {edge} of the long prompt "
        f"{kl['long'][edge]:.2e}); top-1 {agree:.3f}"
    )
    print(summary)
    assert rows.mean() <= MEAN_KL, summary
    assert rows.max() <= MAX_KL, summary
    assert agree >= TOP1, summary
