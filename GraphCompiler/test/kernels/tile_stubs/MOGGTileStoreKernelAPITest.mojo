# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Tile-based kernel stub for testing the TileTensor STORE epilogue fusion
# (OutputTile / OutputFusionTile). Declares a tile `mo.reduce.rms_norm` whose
# output is `_FusedOutputTileTensor`, so the graph compiler selects the store
# path (`_bind_to_fused_output_tile`) for a fused elementwise epilogue. Lives in
# the same library as `TileMatmulStub` (a different op, so no registration
# collision) and drives the store-variant lit test, which only inspects the
# emitted Mojo text (`dump-mojo-text`) -- the body is never executed.
#
from std.collections.string import StaticString
import extensibility
from extensibility import (
    InputTensor,
    _FusedOutputTileTensor,
)
from max.gpu.host import DeviceContext


@extensibility.register("mo.reduce.rms_norm")
struct TileRMSNormStoreStub:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        multiply_before_cast: Bool = True,
    ](
        output: _FusedOutputTileTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Float32,
        weight_offset: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) capturing:
        pass
