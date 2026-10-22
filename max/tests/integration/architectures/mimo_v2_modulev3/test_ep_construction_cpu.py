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
"""The no-GPU gates of MiMo-V2's expert-parallel ModuleV3 path.

The GPU tests stay the numerics gates; everything they depend on that can be
checked without an accelerator is checked here: the parallelism-mode
selection, the EP MoE's whole-expert weight placement under GLOBAL expert
indices (per-shard reindexing would silently load the wrong experts), the
P2P refusal, and the staged trace of the whole EP2 graph on virtual B200s --
the same no-GPU verification lane the TP harness runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from max.driver import (
    CPU,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import (
    Tensor,
    default_device,
    default_dtype,
)
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights, WeightData
from max.nn import kernels
from max.nn.comm.ep import EPBatchManager, EPConfig
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.mimo_v2_modulev3.layers.moe import (
    MiMoV2ExpertParallelMoE,
)
from max.pipelines.architectures.mimo_v2_modulev3.mimo_v2 import (
    MiMoV2,
    ParallelismMode,
    select_parallelism_mode,
)
from max.pipelines.architectures.mimo_v2_modulev3.model import (
    MiMoV2ModuleV3Model,
)
from max.pipelines.architectures.mimo_v2_modulev3.model_config import (
    MiMoV2Config,
)
from max.pipelines.architectures.mimo_v2_modulev3.quant import (
    MXFP8_DISPATCH_QUANT_CONFIG,
)
from max.pipelines.architectures.mimo_v2_modulev3.weight_adapters import (
    convert_safetensor_state_dict,
)
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)
from mimo_v3_runner import kv_params
from tiny_fixture import (
    EXPERTS,
    HIDDEN,
    MOE_DIM,
    TOP_K,
    tensors,
    write,
)
from transformers.configuration_utils import PretrainedConfig

N_DEVICES = 2


def _ep_config(num_devices: int, use_allreduce: bool = True) -> EPConfig:
    """The allreduce EP config the pipeline model builds, tiny-sized."""
    return EPConfig(
        dispatch_dtype=DType.float8_e4m3fn,
        dispatch_quant_config=MXFP8_DISPATCH_QUANT_CONFIG,
        combine_dtype=DType.bfloat16,
        hidden_size=HIDDEN,
        top_k=TOP_K,
        n_experts=EXPERTS,
        max_tokens_per_rank=512,
        n_gpus_per_node=num_devices,
        n_nodes=1,
        use_allreduce=use_allreduce,
    )


def _mxfp4_experts_config() -> QuantConfig:
    """The routed experts' MXFP4 config, as ``parse_quant_scheme`` builds."""
    return QuantConfig.mxfp4(mlp_quantized_layers={1, 2, 3})


# --------------------------------------------------------------------------- #
# Mode selection
# --------------------------------------------------------------------------- #


def test_select_parallelism_mode() -> None:
    ep = _ep_config(N_DEVICES)
    assert select_parallelism_mode(1, 1, None) is ParallelismMode.TP_TP
    assert select_parallelism_mode(2, 1, None) is ParallelismMode.TP_TP
    assert select_parallelism_mode(2, 1, ep) is ParallelismMode.TP_EP
    assert select_parallelism_mode(8, 1, ep) is ParallelismMode.TP_EP
    assert select_parallelism_mode(8, 4, ep) is ParallelismMode.DP_EP


# --------------------------------------------------------------------------- #
# The EP MoE's weight placement
# --------------------------------------------------------------------------- #


def _cpu_mesh() -> DeviceMesh:
    """A CPU mesh standing in for the expert-parallel devices."""
    return DeviceMesh((CPU(),) * N_DEVICES, (N_DEVICES,), ("tp",))


def _ep_moe() -> MiMoV2ExpertParallelMoE:
    """The EP MoE on the tiny fixture's shapes; build it under ``F.lazy``.

    The experts' E8M0 scale tensors compile eagerly on a host backend that
    refuses the dtype, so every construction in this file is lazy.
    """
    return MiMoV2ExpertParallelMoE(
        hidden_dim=HIDDEN,
        num_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        moe_dim=MOE_DIM,
        norm_topk_prob=True,
        quant_config=_mxfp4_experts_config(),
        ep_batch_manager=EPBatchManager(_ep_config(N_DEVICES)),
    )


def test_ep_moe_keeps_global_expert_indices() -> None:
    """The module declares one expert per GLOBAL index, on every layer.

    MAX stacks the experts in-graph; a per-shard reindexing would silently
    load one device's experts into another's slots.
    """
    with F.lazy(), default_dtype(DType.bfloat16):
        layer = _ep_moe()
    names = {name for name, _ in layer.parameters}
    expert_weights = {
        name
        for name in names
        if name.startswith("experts.") and name.endswith(".gate_proj.weight")
    }
    assert expert_weights == {
        f"experts.{i}.gate_proj.weight" for i in range(EXPERTS)
    }


def test_ep_moe_holds_whole_experts_per_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each device's stack covers its contiguous experts at full width.

    Contrast the tensor-parallel stacks, whose axis-1 shard splits every
    expert's channels; the EP stacks keep the whole ``moe_dim`` and cut the
    expert axis instead.
    """
    # The interleave picks its scale layout from the host's architecture.
    monkeypatch.setattr(kernels, "_is_sm10x_gpu", lambda: True)
    with F.lazy(), default_dtype(DType.bfloat16):
        layer = _ep_moe()
        stacked = layer._stacked_weights(_cpu_mesh())
    local = EXPERTS // N_DEVICES
    assert len(stacked) == N_DEVICES
    for gate_up, gate_up_scales, down, down_scales in stacked:
        assert list(gate_up.shape) == [local, 2 * MOE_DIM, HIDDEN // 2]
        assert list(down.shape) == [local, HIDDEN, MOE_DIM // 2]
        # The interleaved weight scales keep the same leading dims.
        assert gate_up_scales.shape[0] == local
        assert down_scales.shape[0] == local


def test_ep_moe_refuses_the_p2p_backend() -> None:
    """MiMo's replicated residual rules the P2P dispatch out (for now)."""
    # The zero-row placeholder below needs a device context, which the
    # virtual-device API provides without a GPU.
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(1)
    with F.lazy(), default_dtype(DType.bfloat16):
        layer = MiMoV2ExpertParallelMoE(
            hidden_dim=HIDDEN,
            num_experts=EXPERTS,
            num_experts_per_tok=TOP_K,
            moe_dim=MOE_DIM,
            norm_topk_prob=True,
            quant_config=_mxfp4_experts_config(),
            ep_batch_manager=EPBatchManager(
                _ep_config(N_DEVICES, use_allreduce=False)
            ),
        )
    with (
        pytest.raises(NotImplementedError, match="allreduce"),
        F.lazy(),
        default_device(DeviceRef.GPU().to_device()),
        default_dtype(DType.bfloat16),
    ):
        # The P2P refusal fires before x is read, so a zero-row lazy
        # placeholder satisfies the signature without ignoring its type.
        layer.forward(Tensor.zeros([]), None)


# --------------------------------------------------------------------------- #
# The whole graph at EP2, staged on virtual B200s
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny fixture checkpoint written to disk, as the GPU tests do."""
    path = tmp_path_factory.mktemp("mimo_v3_ep_cpu")
    write(path, tensors())
    return path


def _hf(checkpoint: Path) -> PretrainedConfig:
    return PretrainedConfig(
        **json.loads((checkpoint / "config.json").read_text())
    )


def _ep_config_model(checkpoint: Path) -> MiMoV2Config:
    """The tiny fixture's config, with the EP config attached."""
    hf = _hf(checkpoint)
    refs = [DeviceRef.GPU(i) for i in range(N_DEVICES)]
    config = MiMoV2Config.from_huggingface_config(
        hf,
        devices=refs,
        kv_params=kv_params(hf, refs, 128),
        max_seq_len=256,
    )
    config.ep_config = _ep_config(N_DEVICES)
    return config


def test_no_ep_init_leaves_the_extra_input_tail_defaulted() -> None:
    """The no-EP path keeps the compile input tail's default.

    ``graph_specs`` reads ``_modulev3_extra_input_types`` on every load,
    and the plain (no-EP) build is the default serve path: the override
    must leave the base's empty-tail default in place when it returns
    early, or every non-EP load fails with an AttributeError.
    """
    model = cast(MiMoV2ModuleV3Model, object.__new__(MiMoV2ModuleV3Model))
    model.pipeline_config = cast(
        Any,
        SimpleNamespace(
            model=SimpleNamespace(data_parallel_degree=1),
            runtime=SimpleNamespace(ep_size=1),
        ),
    )
    model._init_distributed_runtime(cast(Any, None))
    assert model._modulev3_extra_input_types == []


def test_mimo_v2_ep2_stages_with_the_ep_tail(
    monkeypatch: pytest.MonkeyPatch, checkpoint: Path
) -> None:
    """The EP2 graph stages end to end: mode selection, whole-expert MoE
    layers, the EP dispatch/combine ops in the shared MoE subgraph, and the
    EP input tail after the KV leaves."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(N_DEVICES)
    # The trace never runs, so it needs no GPU; the pin makes the kernel
    # wrappers trace the SM100 shape paths the B200 box will compile.
    monkeypatch.setattr(kernels, "_is_sm10x_gpu", lambda: True)

    config = _ep_config_model(checkpoint)
    assert config.ep_config is not None
    ep_config = config.ep_config
    ep_batch_manager = EPBatchManager(ep_config)
    devices = [ref.to_device() for ref in config.devices]
    mesh = DeviceMesh(tuple(devices), (N_DEVICES,), ("tp",))
    weights = SafetensorWeights(sorted(checkpoint.glob("*.safetensors")))
    adapted: dict[str, WeightData] = convert_safetensor_state_dict(
        dict(weights.items()), huggingface_config=_hf(checkpoint)
    )

    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        model = MiMoV2(config, ep_batch_manager)
        inputs = [
            *modulev3_ragged_kv_symbolic_inputs(
                kv_params=config.kv_params,
                device_refs=config.devices,
            ),
            *ep_batch_manager.input_types(),
        ]
        assert model.mode is ParallelismMode.TP_EP
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        staged, _registry = model.stage(*inputs, weights=adapted)
        text = str(staged)

    # The EP tail is declared: the atomic counters plus the three pointer
    # tables per buffer group.
    assert len(ep_batch_manager.input_types()) == 2 * N_DEVICES + 3 * 2
    # The shared MoE subgraph still exists and now carries the EP
    # dispatch/combine; the ops are per-device inside the one body.
    assert text.count("mo.graph @mimo_v2_moe") == 1
    assert text.count("mo.call @mimo_v2_moe") == len(config.moe_layers)
    assert text.count("ep.dispatch") >= N_DEVICES
    assert text.count("ep.combine") >= N_DEVICES
    graphs = set(re.findall(r"mo\.graph @([A-Za-z0-9_]+)", text))
    assert graphs - {"MiMoV2"} == {"mimo_v2_moe"}, graphs


def test_mimo_v2_refuses_dp_and_orphan_ep_managers(
    checkpoint: Path,
) -> None:
    """The DP-attn mode is named but not built, and an EP batch manager
    without an EP config is a wiring error, not a silent TP build."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(N_DEVICES)

    config = _ep_config_model(checkpoint)
    config.data_parallel_degree = N_DEVICES
    mesh = DeviceMesh(
        tuple(ref.to_device() for ref in config.devices), (N_DEVICES,), ("tp",)
    )
    with (
        pytest.raises(NotImplementedError, match="Data-parallel"),
        F.lazy(),
        default_device(mesh),
        default_dtype(config.dtype),
    ):
        MiMoV2(config, None)

    config.data_parallel_degree = 1
    config.ep_config = None
    with (
        pytest.raises(ValueError, match="batch manager"),
        F.lazy(),
        default_device(mesh),
        default_dtype(config.dtype),
    ):
        MiMoV2(config, EPBatchManager(_ep_config(N_DEVICES)))
