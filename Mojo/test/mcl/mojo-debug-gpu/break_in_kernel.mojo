# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# Check that breaking succeeds at kernel launch.

# REQUIRES: NVIDIA-GPU
# FIXME: KERN-1468
# UNSUPPORTED: H100-GPU
# UNSUPPORTED: B200-GPU
# RUN: %mojo-debug-cuda %s | FileCheck %s --check-prefix=CHECK-GDB

from max.gpu import *
from max.gpu.host import DeviceContext

# COM: note the name mangling for PTX.
# CHECK-GDB: break_in_kernel_vec_func


def vec_func(
    output: Pointer[Float32, MutAnyOrigin],
    len: Int,
):
    var tid = global_idx.x
    if tid < len:
        output[tid] = 4.2


def run_vec_func(ctx: DeviceContext) raises:
    comptime length = 1024
    var out_device = ctx.enqueue_create_buffer[.float32](length)
    var block_dim = 32

    ctx.enqueue_function[vec_func](
        out_device,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    # Wait for the kernel to finish running, and give the debugger opportunity to connect.
    ctx.synchronize()


def main() raises:
    with DeviceContext() as ctx:
        run_vec_func(ctx)
