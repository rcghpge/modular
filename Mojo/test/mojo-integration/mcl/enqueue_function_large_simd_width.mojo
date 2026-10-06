# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# REQUIRES: NVIDIA-GPU
# RUN: %mojo %s | FileCheck %s

# COM: MOCO-2418, test that we do not impose wrong alignment for large vector.

from max.gpu.host import DeviceContext


def kernel[
    size: Int,
    dtype: DType,
](x: SIMD[dtype, size], val: Int32):
    print(x)


def main() raises:
    with DeviceContext() as ctx:
        comptime simd_size = 32
        var x = SIMD[.float32, simd_size](0.0)

        comptime kernel_func = kernel[simd_size, DType.float32]
        ctx.enqueue_function[kernel_func](x, Int32(10), grid_dim=1, block_dim=1)
        ctx.synchronize()


# CHECK: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
