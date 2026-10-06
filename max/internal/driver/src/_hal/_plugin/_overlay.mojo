# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from ._trait import DefaultHalPlugin, HalPluginHooks
from .cuda import CudaHalPlugin
from .hip import HipHalPlugin
from .metal import MetalHalPlugin
from std._plugin.selector import PluginSelector

comptime HAL_PLUGINS = PluginSelector[
    DefaultHalPlugin, MetalHalPlugin, CudaHalPlugin, HipHalPlugin
]
