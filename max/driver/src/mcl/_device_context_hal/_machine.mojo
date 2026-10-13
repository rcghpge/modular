# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Staged-tree variant of `_hal/_machine.mojo`.

The `max.gpu.host` overlay stages the `machine` package inside `_hal`, so the
re-exported names resolve relative to it. Staged over `_hal/_machine.mojo` by
`max_srcs_hal` (see `max/driver/src/mcl/BUILD.bazel`); never compiled at
this location.
"""

from .machine import MachineDefinition, DeviceRef, DeviceSpec
