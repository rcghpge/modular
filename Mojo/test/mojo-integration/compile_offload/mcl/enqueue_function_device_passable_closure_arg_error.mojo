# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# `%mojo-build` under asan adds `--sanitize address`; GPU DeviceContext
# codegen then runs for minutes and FileCheck never sees the comptime note.
# UNSUPPORTED: asan

# RUN: not %mojo-build --target-accelerator="nvidia:sm_90a" %s -o %t 2>&1 | FileCheck %s --dump-input=fail

# Compile a kernel against a closure that captures Int32, then enqueue a
# different closure whose capture has an incompatible device type. The
# argument check must still reject that, naming both closure types.
#
# CHECK: constraint failed: argument #0 of type '{{.*}}incompatible_capture{{.*}}' is not encodable as the declared function argument type '{{.*}}expected_capture{{.*}}'

from std.builtin.device_passable import DevicePassable
from max.gpu.host import DeviceContext


def apply_closure[
    closure_type: (def() -> None) & DevicePassable & ImplicitlyCopyable
](closure: closure_type):
    closure()


def main() raises:
    var ctx = DeviceContext()
    var captured_int = Int32(42)
    var captured_simd = SIMD[.int32, 4](1)

    def expected_capture() {var captured_int}:
        _ = captured_int

    def incompatible_capture() {var captured_simd}:
        _ = captured_simd

    comptime kernel = apply_closure[type_of(expected_capture)]
    _ = expected_capture
    var compiled_kernel = ctx.compile_function[kernel]()
    ctx.enqueue_function(
        compiled_kernel,
        incompatible_capture,
        grid_dim=1,
        block_dim=1,
    )
