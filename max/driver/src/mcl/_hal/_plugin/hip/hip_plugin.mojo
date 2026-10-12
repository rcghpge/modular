# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from std.collections.string.string_span import _get_kgen_string
from std.sys.info import CompilationTarget
from .._trait import HalPlugin, HalPluginHooks
from ..._machine import DeviceRef
from ...execution_config._amd import ExecutionConfigForTarget
from ...extensions.hip import HIPHALExtensions


struct HipHalPlugin(HalPlugin):
    """`PluginHooks` implementation for AMD HIP backends.

    Every hook is left at its `PluginHooks` default.
    """

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["hip"]()
    comptime HookFor[target: CompilationTarget] = HipHalPluginHooks[target]


struct HipHalPluginHooks[target: CompilationTarget](HalPluginHooks):
    comptime ExecutionConfigType[device: DeviceRef] = ExecutionConfigForTarget[
        device.spec.name
    ]
    """Returns the execution configuration for the given device."""

    comptime extensions = HIPHALExtensions
