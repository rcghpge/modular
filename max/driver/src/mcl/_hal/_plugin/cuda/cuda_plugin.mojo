# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from std.collections.string.string_span import _get_kgen_string
from .._trait import HalPluginHooks
from ..._machine import DeviceRef
from ...execution_config._nvidia import ExecutionConfigForTarget
from ...extensions.cuda import CUDAHALExtensions


struct CudaHalPlugin(HalPluginHooks):
    """`PluginHooks` implementation for NVIDIA CUDA backends.

    Every hook is left at its `PluginHooks` default.
    """

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["cuda"]()

    comptime ExecutionConfigType[device: DeviceRef] = ExecutionConfigForTarget[
        device.spec.name
    ]
    """Returns the execution configuration for the given device."""

    comptime extensions = CUDAHALExtensions
