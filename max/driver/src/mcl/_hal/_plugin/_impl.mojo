# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Selects the active HAL plugin hooks.

`CurrentHalPlugin` is resolved from the registered `HAL_PLUGINS`, keyed on the
target's `stdlib_plugin` field. The default build resolves to
`DefaultHalPlugin`.
"""

from ._trait import HalPluginHooks
from ._overlay import HAL_PLUGINS
from std.sys.info import CompilationTarget

comptime CurrentHalPlugin: HalPluginHooks = HAL_PLUGINS.current.HookFor[
    CompilationTarget.current()
]
"""The active HAL plugin."""
