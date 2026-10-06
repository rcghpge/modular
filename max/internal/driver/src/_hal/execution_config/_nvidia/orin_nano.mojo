# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Execution config placeholder for the NVIDIA Jetson Orin Nano (Ampere embedded) GPU."""

from max.gpu.host import ConstantMemoryMapping, Dim, LaunchAttribute
from . import (
    LaunchAttributeHolderExecutionConfig,
    _CudaExecutionConfigDelegator,
    _BaseCudaExecutionConfig,
    _BaseCudaExecutionConfiguration,
)
from .. import (
    ExecutionConfig,
    BlockExecutionConfig,
    GridBlockExecutionConfig,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
)


struct OrinNanoExecutionConfiguration(
    BlockExecutionConfig,
    ConstantMemoryMappingExecutionConfig,
    Copyable,
    ExecutionConfig,
    GridBlockExecutionConfig,
    LaunchAttributeHolderExecutionConfig,
    Movable,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
    _BaseCudaExecutionConfig,
    _CudaExecutionConfigDelegator,
):
    """Describes the execution configuration of a GPU kernel launch on the
    NVIDIA Jetson Orin Nano."""

    var _base: _BaseCudaExecutionConfiguration

    def __init__(
        out self,
        *,
        var block_dim: Dim,
        var grid_dim: Dim,
        var shared_mem_bytes: Int = 0,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
    ):
        """The 'all members' constructor with reasonable defaults set."""
        self._base = _BaseCudaExecutionConfiguration(
            block_dim=block_dim,
            grid_dim=grid_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    def _get_inner_config[
        o: Origin
    ](ref[o] self) -> ref[o] _BaseCudaExecutionConfiguration:
        """Gets the inner `_BaseCudaExecutionConfiguration` to which this delegator forwards
        all calls.

        Returns:
            The inner `_BaseCudaExecutionConfiguration`.
        """
        return Pointer(to=self._base).unsafe_origin_cast[o]()[]
