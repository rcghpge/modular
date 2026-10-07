# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Machine Definition and Static Device Information.

Provides structures for defining, specifying, and deriving
compilation info for computational resources on the current
(native) or another (cross-compile) machine."""

from .device import DeviceRef, DeviceSpec
from .machine import MachineDefinition
