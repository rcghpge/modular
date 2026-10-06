# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-build --target-accelerator="nvidia:sm_90a" %s -o %t

from std.builtin.device_passable import DevicePassable
from max.gpu.host import DeviceContext


def call_closure[
    closure_type: (def() -> None) & DevicePassable & ImplicitlyCopyable
](closure: closure_type):
    closure()


def main() raises:
    var ctx = DeviceContext()
    var captured = Int32(42)

    def closure() {var captured}:
        _ = captured

    comptime kernel = call_closure[type_of(closure)]
    var compiled_kernel = ctx.compile_function[kernel]()
    ctx.enqueue_function(
        compiled_kernel,
        closure,
        grid_dim=1,
        block_dim=1,
    )
