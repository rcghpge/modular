# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Tile-based kernel stubs for testing TileTensor epilogue fusion (GEX-3747).
# Kept separate from testing_kernels so that tests using the general
# kernel-library-paths substitution are not affected by mo.matmul being
# overridden with a tile-output variant.
#
from std.collections.string import StaticString
import extensibility
from extensibility import (
    InputTensor,
    _FusedComputeOutputTileTensor,
)
from max.gpu.host import DeviceContext


@extensibility.register("mo.matmul")
struct TileMatmulStub:
    @staticmethod
    def execute[
        transpose_b: Bool,
        packed_b: Bool,
        has_epilogue_fusion: Bool,
        target: StaticString,
        _trace_name: StaticString,
    ](
        c: _FusedComputeOutputTileTensor[rank=2, ...],
        a: InputTensor[rank=2, ...],
        b: InputTensor[rank=2, ...],
        ctx: DeviceContext,
    ) capturing:
        pass
