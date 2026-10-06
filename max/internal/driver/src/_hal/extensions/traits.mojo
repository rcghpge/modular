# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from std.ffi import _DLHandle

from ..plugins import HALPluginConstructable
from ..status import (
    STATUS_SUCCESS,
    STATUS_UNKNOWN_ERROR,
    STATUS_UNKNOWN_PROPERTY_NAME,
    STATUS_UNKNOWN_SYMBOL,
    HALError,
)


trait HALExtensions(Deinitable, HALPluginConstructable, Movable):
    """Compile-time interface for a vendor's HAL extension set.

    Each backend provides a struct conforming to this trait to expose its
    surface of HAL extension functions. A plugin advertises its extension set
    via the `comptime extensions` member on `HalPluginHooks`.
    """

    comptime SupportsGraphCaptureV1: Bool = conforms_to(
        Self, SupportsGraphCaptureV1
    )

    def __init__(
        out self, lib: _DLHandle, so_path: StringSlice
    ) raises HALError:
        ...


trait SupportsGraphCaptureV1(Deinitable, Movable):
    comptime GraphCaptureTy: GraphCaptureV1

    def graph_capture(ref self) -> ref[self] Self.GraphCaptureTy:
        ...


trait GraphCaptureV1:
    def start_capture(mut self) raises:
        ...

    def end_capture(mut self) raises:
        ...


struct DefaultHALExtensions(HALExtensions):
    """`HALExtensions` implementation used when no plugin is active."""

    def __init__(
        out self, lib: _DLHandle, so_path: StringSlice
    ) raises HALError:
        pass
