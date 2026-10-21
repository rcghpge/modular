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
"""MiMo-V2 routed experts with MXFP8 activations, on ModuleV3.

The routed experts run the same W4A8 chain as the graph-API architecture
(:class:`max.nn.moe.MoEQuantized` on NVIDIA), through
:class:`max.nn.moe.NvMxf4f8Strategy`; the modulev3 MoE layers have no
MXFP4/e8m0 path of their own. Each device holds its share of every expert's
channels, so its chain yields a partial sum the residual add reduces.
"""

from __future__ import annotations

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers import quant_ops
from max.experimental.nn.common_layers.functional_kernels import (
    moe_router_group_limited,
)
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.common_layers.quant_tensor import FP8BlockTensor
from max.experimental.nn.sequential import ModuleList
from max.experimental.realization_context import ensure_context
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    NamedMapping,
    Partial,
)
from max.experimental.tensor import Tensor
from max.graph import TensorValue, ops
from max.nn.activation import GatedActivation
from max.nn.comm.ep.ep_kernels import fused_silu
from max.nn.kernels import moe_create_indices, moe_finalize
from max.nn.moe import NvMxf4f8Strategy
from max.nn.quant_config import QuantConfig

from ..quant import FP8_BLOCK


class _Fp8Weights(Module[[], None]):
    """One dense FP8 projection's weight and 128x128 block scales.

    Both stay on the host as stored; the graph gives each device its share.
    """

    def __init__(self, out_dim: int, in_dim: int) -> None:
        self.weight = Tensor.zeros(
            [out_dim, in_dim], dtype=DType.float8_e4m3fn, device=CPU()
        )
        self.weight_scale = Tensor.zeros(
            [out_dim // FP8_BLOCK, in_dim // FP8_BLOCK],
            dtype=DType.float32,
            device=CPU(),
        )

    def matmul(self, x: Tensor, axis: int) -> Tensor:
        """Returns ``x @ weight.T``, the weight sharded on ``axis``.

        A split on the weight's rows (``axis == 0``) is column-parallel and
        shards the output's columns; one on its columns leaves each device a
        partial sum of the output.
        """
        quant = FP8BlockTensor(
            data=self.weight,
            weight_scale_inv=self.weight_scale,
            block_size=(FP8_BLOCK, FP8_BLOCK),
        )
        return quant_ops.matmul(x, quant.shard(axis, x.mesh))


class MiMoV2MLP(Module[[Tensor], Tensor]):
    """The dense FP8 SwiGLU MLP of the one non-MoE layer: ``down(silu(gate)
    * up)``."""

    def __init__(self, *, hidden_dim: int, intermediate_dim: int) -> None:
        self.gate_proj = _Fp8Weights(intermediate_dim, hidden_dim)
        self.up_proj = _Fp8Weights(intermediate_dim, hidden_dim)
        self.down_proj = _Fp8Weights(hidden_dim, intermediate_dim)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate_proj.matmul(x, axis=0)
        up = self.up_proj.matmul(x, axis=0)
        return self.down_proj.matmul(F.silu(gate) * up, axis=1)


class MiMoV2Router(Module[[Tensor], tuple[Tensor, Tensor]]):
    """Sigmoid top-k router whose logits are computed in float32.

    The reference scores with ``F.linear(hidden.float(), weight.float())``
    on the BF16 gate weight; the weight's cast runs once, at model init. The
    correction bias steers selection only; the combine weights are the
    unbiased sigmoid scores renormalized over the top k.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool,
    ) -> None:
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.gate_score = Linear(hidden_dim, num_experts, bias=False)
        self.e_score_correction_bias = Tensor.zeros(
            [num_experts], dtype=DType.float32
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mesh = x.mesh
        if mesh.num_devices <= 2:
            # Every device routes its own replica of the tokens. The replicas
            # come out of an allreduce, and a two-way sum is the same in
            # either order, so both devices pick the same experts without a
            # collective.
            return self._route(
                x, self.gate_score.weight, self.e_score_correction_bias
            )
        # A wider allreduce sums in a different order on each device, so a
        # replica can differ in its last bit and break a near tie the other
        # way. Device 0 routes for every device.
        experts, weights = self._route(
            x.local_shards[0],
            self.gate_score.weight.local_shards[0],
            self.e_score_correction_bias.local_shards[0],
        )
        return (
            F.distributed_broadcast(experts, mesh),
            F.distributed_broadcast(weights, mesh),
        )

    def _route(
        self, x: Tensor, weight: Tensor, bias: Tensor
    ) -> tuple[Tensor, Tensor]:
        logits = x.cast(DType.float32) @ weight.cast(DType.float32).T
        scores = F.sigmoid(logits)
        return moe_router_group_limited(
            scores,
            bias,
            self.num_experts,
            self.num_experts_per_tok,
            n_groups=1,
            topk_group=1,
            norm_weights=self.norm_topk_prob,
            routed_scaling_factor=1.0,
        )


class _MxWeights(Module[[], None]):
    """One routed expert's MXFP4 projection, as the checkpoint stores it.

    Packed E2M1 codes with one E8M0 scale per 32 inputs; the stacks stay on
    the host, and the graph cuts each device's share.
    """

    def __init__(self, out_dim: int, in_dim: int) -> None:
        self.weight = Tensor.zeros(
            [out_dim, in_dim // 2], dtype=DType.uint8, device=CPU()
        )
        self.weight_scale = Tensor.zeros(
            [out_dim, in_dim // 32],
            dtype=DType.float8_e8m0fnu,
            device=CPU(),
        )


class MiMoV2Expert(Module[[], None]):
    """One routed expert's gate, up and down projections."""

    def __init__(self, moe_dim: int, hidden_dim: int) -> None:
        self.gate_proj = _MxWeights(moe_dim, hidden_dim)
        self.up_proj = _MxWeights(moe_dim, hidden_dim)
        self.down_proj = _MxWeights(hidden_dim, moe_dim)


class MiMoV2MoE(Module[[Tensor], Tensor]):
    """MiMo-V2's sigmoid-routed MoE, with no shared expert.

    On NVIDIA, the experts run W4A8: each projection's input is quantized to
    MXFP8 per expert at runtime. Every device runs every routed row through
    its share of each expert's channels, which gives a partial sum of the
    output the residual add reduces.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_dim: int,
        norm_topk_prob: bool,
        quant_config: QuantConfig,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_dim = moe_dim
        self.activation = GatedActivation()
        self.quant_config = quant_config
        self.gate = MiMoV2Router(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
        )
        self.experts = ModuleList(
            [MiMoV2Expert(moe_dim, hidden_dim) for _ in range(num_experts)]
        )
        # MXFP4 inputs give about 5x the expert error of MXFP8 against a
        # float32 reference (test_expert_precision_gpu.py), so the experts'
        # activations stay MXFP8.
        self.strategy = NvMxf4f8Strategy(quant_config, DType.float8_e4m3fn)

    def _stacked_weights(
        self, mesh: DeviceMesh
    ) -> list[tuple[TensorValue, ...]]:
        """Each device's gate/up and down stacks, with interleaved scales.

        Reads only weights, so the graph compiler runs it once, at model
        init. The gate/up stacks pair each expert's gate rows with its up
        rows, as the unfused gated activation splits them.
        """
        e, md, hidden = self.num_experts, self.moe_dim, self.hidden_dim
        n = mesh.num_devices
        gate = F.stack([x.gate_proj.weight for x in self.experts], axis=0)
        up = F.stack([x.up_proj.weight for x in self.experts], axis=0)
        down = F.stack([x.down_proj.weight for x in self.experts], axis=0)
        gate_scale = F.stack(
            [x.gate_proj.weight_scale for x in self.experts], axis=0
        )
        up_scale = F.stack(
            [x.up_proj.weight_scale for x in self.experts], axis=0
        )
        down_scale = F.stack(
            [x.down_proj.weight_scale for x in self.experts], axis=0
        )
        # Each device's share of every expert's channels: rows of the gate
        # and up projections, columns of the down projection.
        gate = gate.to(NamedMapping(mesh, (None, TP, None)))
        up = up.to(NamedMapping(mesh, (None, TP, None)))
        gate_scale = gate_scale.to(NamedMapping(mesh, (None, TP, None)))
        up_scale = up_scale.to(NamedMapping(mesh, (None, TP, None)))
        down = down.to(NamedMapping(mesh, (None, None, TP)))
        down_scale = down_scale.to(NamedMapping(mesh, (None, None, TP)))
        per_device: list[tuple[TensorValue, ...]] = []
        for d in range(n):
            gate_up = ops.stack(
                [
                    TensorValue(gate.local_shards[d]),
                    TensorValue(up.local_shards[d]),
                ],
                axis=1,
            ).reshape([e, 2 * md // n, hidden // 2])
            gate_up_scales = ops.stack(
                [
                    TensorValue(gate_scale.local_shards[d]),
                    TensorValue(up_scale.local_shards[d]),
                ],
                axis=1,
            ).reshape([e, 2 * md // n, hidden // 32])
            down_w = TensorValue(down.local_shards[d])
            gate_up_interleaved, down_interleaved = (
                self.strategy.prepare_weight_scales(
                    gate_up_scales,
                    TensorValue(down_scale.local_shards[d]),
                    down_w.device,
                )
            )
            per_device.append(
                (gate_up, gate_up_interleaved, down_w, down_interleaved)
            )
        return per_device

    def _experts_on_device(
        self,
        x: TensorValue,
        router_idx: TensorValue,
        router_weight: TensorValue,
        gate_up: TensorValue,
        gate_up_scales: TensorValue,
        down: TensorValue,
        down_scales: TensorValue,
    ) -> TensorValue:
        """Runs the W4A8 expert chain on one device."""
        strategy = self.strategy
        group = self.quant_config.input_scale.block_size
        assert group is not None
        (
            token_order,
            expert_start,
            restore_order,
            expert_ids,
            usage_stats,
            scales_offset,
        ) = moe_create_indices(
            ops.cast(ops.reshape(router_idx, [-1]), DType.int32),
            self.num_experts,
            needs_scales_offset=True,
        )
        total_m = ops.shape_to_tensor(token_order.shape)[0].cast(DType.uint32)
        # The grouped quantize gathers rows through the slot order itself,
        # dividing by the top-k in its loads, so neither the token indices
        # nor the permuted activations materialize.
        permuted_quant, permuted_scales = strategy.grouped_quantize(
            x,
            group[1],
            None,
            expert_start,
            scales_offset,
            expert_ids,
            indices=token_order,
            indices_divisor=self.num_experts_per_tok,
        )
        gate_up_out = strategy.grouped_matmul(
            gate_up,
            gate_up_scales,
            expert_inputs=(
                permuted_quant,
                permuted_scales,
                expert_start,
                scales_offset,
                expert_ids,
                usage_stats,
            ),
            estimated_total_m=total_m,
        )
        gate_up_out = fused_silu(
            gate_up_out,
            expert_start,
            activation=self.activation,
            interleaved=False,
        )
        gate_up_quant, gate_up_scales = strategy.grouped_quantize(
            gate_up_out, group[1], None, expert_start, scales_offset, expert_ids
        )
        down_out = strategy.grouped_matmul(
            down,
            down_scales,
            expert_inputs=(
                gate_up_quant,
                gate_up_scales,
                expert_start,
                scales_offset,
                expert_ids,
                usage_stats,
            ),
            estimated_total_m=total_m,
        )
        return moe_finalize(down_out, restore_order, router_weight, x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        experts, weights = self.gate(x)
        mesh = x.mesh
        n = mesh.num_devices
        stacked = self._stacked_weights(mesh)
        with ensure_context():
            shards = [
                self._experts_on_device(
                    TensorValue(x.local_shards[d]),
                    TensorValue(experts.local_shards[d]),
                    TensorValue(weights.local_shards[d]),
                    *stacked[d],
                )
                for d in range(n)
            ]
        if n == 1:
            return Tensor.from_shard_values(shards, x.mapping)
        return Tensor.from_shard_values(
            shards, DeviceMapping(mesh, (Partial(),))
        )
