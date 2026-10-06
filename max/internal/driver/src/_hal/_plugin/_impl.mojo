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

comptime CurrentHalPlugin: HalPluginHooks = HAL_PLUGINS.current
"""The active HAL plugin."""
