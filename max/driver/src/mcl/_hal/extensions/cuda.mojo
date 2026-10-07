# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from std.ffi import _DLHandle

from ..plugins import RawPluginComponent
from ..status import HALError
from .traits import HALExtensions


struct CUDAExtensionFunctions(RawPluginComponent):
    pass


struct CUDAHALExtensions(HALExtensions):
    """`HALExtensions` implementation for NVIDIA CUDA backends."""

    def __init__(
        out self, lib: _DLHandle, so_path: StringSlice
    ) raises HALError:
        pass
