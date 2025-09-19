# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from math import ceildiv

import gpu.warp as warp
from gpu import WARP_SIZE
from gpu.host import DeviceContext
from linalg.gemv import gemv_kernel
from linalg.matmul_gpu import matmul_kernel_naive
from testing import assert_false

from utils.numerics import isnan
from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from layout.runtime_layout import RuntimeLayout
from utils.index import IndexList


fn run_matvec[
    reduction_method: warp.ReductionMethod
](M: Int, N: Int, K: Int, *, ctx: DeviceContext) raises:
    print("== run_matvec kernel")

    var iterations = 100
    var a_host = UnsafePointer[BFloat16].alloc(M * K)
    var b_host = UnsafePointer[BFloat16].alloc(K * N)
    var c_host = UnsafePointer[Float32].alloc(M * N)
    var a_host_n = UnsafePointer[Float32].alloc(M * K)
    var b_host_n = UnsafePointer[Float32].alloc(K * N)
    var c_host_n = UnsafePointer[Float32].alloc(M * N)

    for i in range(M * K):
        a_host[i] = i
        a_host_n[i] = i

    for i in range(K * N):
        b_host[i] = i + 1
        b_host_n[i] = i + 1

    for i in range(M * N):
        c_host[i] = 0

    for i in range(M * N):
        c_host_n[i] = 0

    var a_device = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var b_device = ctx.enqueue_create_buffer[DType.bfloat16](K * N)
    var c_device = ctx.enqueue_create_buffer[DType.float32](M * N)
    var a_device_n = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b_device_n = ctx.enqueue_create_buffer[DType.float32](K * N)
    var c_device_n = ctx.enqueue_create_buffer[DType.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    alias WARPS_PER_BLOCK = 32
    alias kernel = gemv_kernel[
        DType.float32,
        DType.bfloat16,
        DType.bfloat16,
        reduction_method=reduction_method,
    ]

    @always_inline
    @parameter
    fn run_func_gemv(ctx: DeviceContext) raises:
        ctx.enqueue_function_checked[kernel, kernel](
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=ceildiv(M, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
        )

    var kernelType = "GEMV"
    var nstime = ctx.execution_time[run_func_gemv](iterations)
    var flops = 2 * M * N * K
    var sectime = (nstime / iterations) / 1000000000
    print(kernelType, "KERNEL:")
    print(sectime, "sec")
    print(flops * 1e-9 / sectime, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host, c_device)

    # running naive
    ctx.enqueue_copy(a_device_n, a_host_n)
    ctx.enqueue_copy(b_device_n, b_host_n)

    alias BLOCK_DIM = 16

    # Create layout tensors for the naive kernel
    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var c_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        c_device_n._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, N)),
    )

    var a_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        a_device_n._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, K)),
    )

    var b_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        b_device_n._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](K, N)),
    )

    @always_inline
    @parameter
    fn run_func_naive(ctx: DeviceContext) raises:
        alias kernel = matmul_kernel_naive[
            DType.float32,
            DType.float32,
            DType.float32,
            c_tensor.layout,
            a_tensor.layout,
            b_tensor.layout,
            BLOCK_DIM,
        ]

        ctx.enqueue_function_checked[kernel, kernel](
            c_tensor,
            a_tensor,
            b_tensor,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

    nstime = ctx.execution_time[run_func_naive](iterations)
    var sectime2 = (nstime / iterations) / 1000000000
    print("SHMEM MATMUL:")
    print(sectime2, "sec")
    print(flops * 1e-9 / sectime2, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host_n, c_device_n)
    ctx.synchronize()

    # Due to varied pattern of FP arith the accumulated sum isn't exactly
    # accurate. Hence relative tolerance needs to be checked.
    alias errorTolerance = 0.1
    var failed = False
    for i in range(M * N):
        var outVal = c_host[i]
        var outRef = c_host_n[i]
        var relDiff = (max(outVal, outRef) / min(outVal, outRef)) - 1.0
        if (relDiff > errorTolerance) or isnan(outVal) or isnan(outRef):
            failed = True

    if not failed:
        print("Success 🎉: results match")
        print(
            "Performance warp-shuffle matvec vs. shmem matmul: ",
            sectime2 / sectime,
            "x",
        )
    else:
        print("Failed ❌: results mismatch")

    assert_false(failed)

    _ = a_device
    _ = b_device
    _ = c_device

    _ = a_device_n
    _ = b_device_n
    _ = c_device_n

    _ = a_host
    _ = b_host
    _ = c_host

    _ = a_host_n
    _ = b_host_n
    _ = c_host_n


def main():
    with DeviceContext() as ctx:
        run_matvec[reduction_method = warp.ReductionMethod.WARP](
            4096, 1, 4096, ctx=ctx
        )
        run_matvec[reduction_method = warp.ReductionMethod.TENSOR_CORE](
            4096, 1, 4096, ctx=ctx
        )
