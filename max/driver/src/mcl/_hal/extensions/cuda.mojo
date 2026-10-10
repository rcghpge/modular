# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from std.collections import OptionalReg
from std.ffi import _DLHandle

from ..plugins import RawPluginComponent
from ..status import HALError
from .peer_access import RawPluginPeerAccess
from .traits import HALExtensions, SupportsPeerAccessV1


struct CUDAExtensionFunctions(RawPluginComponent):
    pass


struct CUDAHALExtensions(HALExtensions, SupportsPeerAccessV1):
    """`HALExtensions` implementation for NVIDIA CUDA backends."""

    comptime PeerAccessTy = RawPluginPeerAccess
    var _peer: OptionalReg[RawPluginPeerAccess]

    def __init__(
        out self, lib: _DLHandle, so_path: StringSlice
    ) raises HALError:
        self._peer = RawPluginPeerAccess.load(lib, so_path)

    def peer_access(self) -> OptionalReg[Self.PeerAccessTy]:
        return self._peer
