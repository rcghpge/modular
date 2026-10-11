# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .._trait import HalPluginHooks
from ..._machine import DeviceRef
from std.collections.string.string_span import _get_kgen_string
from ...execution_config._apple import ExecutionConfigForTarget
from ...extensions.metal import MetalHALExtensions


struct MetalHalPlugin(HalPluginHooks):
    """`PluginHooks` implementation for Apple Metal backends.

    Every hook is left at its `PluginHooks` default.
    """

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["metal"]()

    comptime ExecutionConfigType[device: DeviceRef] = ExecutionConfigForTarget[
        device.spec.name
    ]
    """Returns the execution configuration for the given device."""

    comptime extensions = MetalHALExtensions
