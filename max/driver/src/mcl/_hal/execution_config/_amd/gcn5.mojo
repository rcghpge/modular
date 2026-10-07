# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Execution config placeholder for AMD GCN 5th generation (Vega) GPUs:
gfx900 (Radeon Vega Frontier Edition, Radeon RX Vega 64), gfx902 (Ryzen APU,
e.g. Ryzen 3 2200G, Ryzen 5 2400G), gfx904, gfx906 (Radeon VII, Radeon
Instinct MI50). Predates the CDNA split (gfx908 and later)."""

from max.gpu.host import ConstantMemoryMapping, Dim
from . import (
    _BaseAmdExecutionConfiguration,
    _BaseAmdExecutionConfig,
    _AmdExecutionConfigDelegator,
)
from .. import (
    ExecutionConfig,
    BlockExecutionConfig,
    GridBlockExecutionConfig,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
)


struct GCN5ExecutionConfiguration(
    BlockExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
    Copyable,
    ExecutionConfig,
    GridBlockExecutionConfig,
    Movable,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
    _AmdExecutionConfigDelegator,
    _BaseAmdExecutionConfig,
):
    """Describes the execution configuration of a GPU kernel launch on AMD
    GCN 5th generation GPUs."""

    var _base: _BaseAmdExecutionConfiguration

    def __init__(
        out self,
        *,
        var block_dim: Dim,
        var grid_dim: Dim,
        var shared_mem_bytes: Int = 0,
        var constant_memory: List[ConstantMemoryMapping] = [],
    ):
        """The 'all members' constructor with reasonable defaults set."""
        self._base = _BaseAmdExecutionConfiguration(
            block_dim=block_dim,
            grid_dim=grid_dim,
            shared_mem_bytes=shared_mem_bytes,
            constant_memory=constant_memory^,
        )

    def _get_inner_config[
        o: Origin
    ](ref[o] self) -> ref[o] _BaseAmdExecutionConfiguration:
        """Gets the inner `_BaseAmdExecutionConfiguration` to which this delegator forwards
        all calls.

        Returns:
            The inner `_BaseAmdExecutionConfiguration`.
        """
        return Pointer(to=self._base).unsafe_origin_cast[o]()[]
