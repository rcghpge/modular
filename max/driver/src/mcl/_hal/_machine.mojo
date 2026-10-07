# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Re-exports the `machine` names `_hal` uses.

Here `machine` is a sibling top-level package; in the staged stdlib overlay it
is nested inside `_hal`, and this file is shadowed by a variant importing from
that location (see Mojo/stdlib/std/BUILD.bazel). `_hal` sources import these
names relatively (`from ._machine import ...`), which resolves in both
compilation contexts.
"""

from machine import MachineDefinition, DeviceRef, DeviceSpec
