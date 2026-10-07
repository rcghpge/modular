# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from std.ffi import _DLHandle

from ..plugins import RawPluginComponent
from ..status import HALError
from .traits import HALExtensions


struct MetalExtensionFunctions(RawPluginComponent):
    pass


struct MetalHALExtensions(HALExtensions):
    """`HALExtensions` implementation for Apple Metal backends."""

    def __init__(
        out self, lib: _DLHandle, so_path: StringSlice
    ) raises HALError:
        pass
