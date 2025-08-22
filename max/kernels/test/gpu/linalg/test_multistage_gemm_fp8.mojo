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


import linalg.vendor_blas
from buffer.dimlist import DimList
from gpu import grid_dim
from gpu.host import DeviceContext, FuncAttribute
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    zero,
)
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.layout import *
from linalg._multistage_gemm_gpu import multistage_gemm_kernel
from linalg.utils_gpu import (
    MatmulKernels,
)


fn test_fp8_multistage_gemm[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    /,
    *,
    transpose_b: Bool = False,
](ctx: DeviceContext) raises:
    print("test fp8 multistage matmul")

    alias static_a_shape = DimList(M, K)
    alias static_b_shape = DimList(N, K) if transpose_b else DimList(K, N)
    alias static_c_shape = DimList(M, N)

    var a_host = HostNDBuffer[dtype, 2, static_a_shape]()
    var b_host = HostNDBuffer[dtype, 2, static_b_shape]()
    var c_host = HostNDBuffer[DType.float32, 2, static_c_shape]()
    var c_host_ref = HostNDBuffer[DType.float32, 2, static_c_shape]()

    @parameter
    for i in range(M):

        @parameter
        for j in range(K):
            a_host.tensor[i, j] = i + j

    @parameter
    for i in range(static_b_shape.get[0]()):

        @parameter
        for j in range(static_b_shape.get[1]()):
            b_host.tensor[i, j] = i + j

    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    var a_device = DeviceNDBuffer[dtype, 2, static_a_shape](ctx=ctx)
    var b_device = DeviceNDBuffer[dtype, 2, static_b_shape](ctx=ctx)
    var c_device = DeviceNDBuffer[DType.float32, 2, static_c_shape](ctx=ctx)
    var c_device_ref = DeviceNDBuffer[DType.float32, 2, static_c_shape](ctx=ctx)

    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

    var c_tensor = from_ndbuffer_row_major(c_device.tensor)
    var a_tensor = from_ndbuffer_row_major(a_device.tensor)
    var b_tensor = from_ndbuffer_row_major(b_device.tensor)

    alias kernels = MatmulKernels[dtype, dtype, DType.float32, transpose_b]()
    alias config = kernels.hopper_128x128_4

    alias kernel = multistage_gemm_kernel[
        DType.float32,  # c_type
        c_tensor.layout,
        dtype,  # a_type
        a_tensor.layout,
        dtype,  # b_type
        b_tensor.layout,
        transpose_b,
        config,
    ]

    alias BM = config.block_tile_shape[0]
    alias BN = config.block_tile_shape[1]

    ctx.enqueue_function[kernel](
        c_tensor,
        a_tensor,
        b_tensor,
        grid_dim=config.grid_dim(M, N),
        block_dim=config.block_dim(),
        shared_mem_bytes=config.shared_mem_usage(),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            config.shared_mem_usage()
        ),
    )

    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)

    if transpose_b:
        vendor_blas.matmul(
            ctx,
            c_device_ref.tensor,
            a_device.tensor,
            b_device.tensor,
            c_row_major=True,
        )

    else:
        # TODO: Matrix B should always be in col-major layout for cublasLt to work
        var b_host_col_major = HostNDBuffer[dtype, 2, DimList(N, K)]()

        for i in range(N):
            for j in range(K):
                b_host_col_major.tensor[i, j] = b_host.tensor[j, i]

        var b_device_col_major = DeviceNDBuffer[dtype, 2, DimList(N, K)](
            ctx=ctx
        )
        ctx.enqueue_copy(
            b_device_col_major.buffer, b_host_col_major.tensor.data
        )

        vendor_blas.matmul(
            ctx,
            c_device_ref.tensor,
            a_device.tensor,
            b_device_col_major.tensor,
            c_row_major=True,
        )

    ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)

    ctx.synchronize()

    assert_almost_equal(
        c_host.tensor,
        c_host_ref.tensor,
        atol=0.0001,
        rtol=0.01,
    )

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device

    _ = a_tensor
    _ = b_tensor
    _ = c_tensor


def main():
    with DeviceContext() as ctx:
        test_fp8_multistage_gemm[
            DType.float8_e4m3fn, 128, 128, 64, transpose_b=True
        ](ctx)
        test_fp8_multistage_gemm[
            DType.float8_e4m3fn, 128, 128, 128, transpose_b=True
        ](ctx)
        test_fp8_multistage_gemm[
            DType.float8_e4m3fn, 128, 128, 64, transpose_b=False
        ](ctx)
        test_fp8_multistage_gemm[
            DType.float8_e4m3fn, 128, 128, 128, transpose_b=False
        ](ctx)
