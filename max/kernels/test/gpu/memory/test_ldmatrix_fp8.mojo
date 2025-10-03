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

from gpu import WARP_SIZE, barrier, lane_id
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from gpu.mma import ld_matrix, mma
from gpu.mma_util import store_matrix_d
from layout import UNKNOWN_VALUE, Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from layout.tensor_core import get_fragment_size, get_mma_shape
from linalg.matmul.gpu import matmul_kernel_naive
from memory import stack_allocation
from testing import assert_almost_equal

from utils import IndexList
from utils.numerics import get_accum_type


fn test_ldmatrix_fp8[
    input_type: DType,
](
    c_ptr: UnsafePointer[Float32],
    a_ptr: UnsafePointer[Scalar[input_type]],
    b_ptr: UnsafePointer[Scalar[input_type]],
):
    alias accum_type = get_accum_type[input_type]()
    alias mma_shape = get_mma_shape[input_type, accum_type]()
    alias M = mma_shape[0]
    alias N = mma_shape[1]
    alias K = mma_shape[2]
    alias frag_size = get_fragment_size[mma_shape]()
    alias a_frag_size = frag_size[0]
    alias b_frag_size = frag_size[1]
    alias c_frag_size = frag_size[2]

    var d = SIMD[accum_type, c_frag_size](0)

    var a_shared = stack_allocation[
        M * K, input_type, alignment=32, address_space = AddressSpace.SHARED
    ]()
    var b_shared = stack_allocation[
        N * K, input_type, alignment=32, address_space = AddressSpace.SHARED
    ]()

    for i in range(lane_id(), M * K, WARP_SIZE):
        a_shared[i] = a_ptr[i]

    for i in range(lane_id(), N * K, WARP_SIZE):
        b_shared[i] = b_ptr[i]

    barrier()

    var a_reg = ld_matrix[a_frag_size](
        a_shared + Int((lane_id() % 16) * 32 + (lane_id() // 16) * 16)
    )

    var b_reg = ld_matrix[b_frag_size](
        b_shared + Int((lane_id() % 8) * 32 + (lane_id() // 8) * 16)
    )

    mma(d, a_reg, b_reg, d)
    store_matrix_d[M, N, K](
        c_ptr,
        # Store matrix is hardcoded to store 4 elements.
        rebind[SIMD[DType.float32, 4]](d),
        0,
        0,
        mma_shape[1],
    )


fn check_ldmatrix_fp8[
    input_type: DType,
](ctx: DeviceContext) raises:
    print("== test ldmatrix transposed fp8")

    # Shape for a single mma.
    alias accum_type = get_accum_type[input_type]()
    alias mma_shape = get_mma_shape[input_type, accum_type]()
    alias M = mma_shape[0]
    alias N = mma_shape[1]
    alias K = mma_shape[2]

    var a_host = UnsafePointer[Scalar[input_type]].alloc(M * K)
    var b_host = UnsafePointer[Scalar[input_type]].alloc(K * N)
    var c_host = UnsafePointer[Float32].alloc(M * N)
    var c_host_ref = UnsafePointer[Float32].alloc(M * N)

    @parameter
    for m in range(M):

        @parameter
        for k in range(K):
            a_host[m * K + k] = m + k

    @parameter
    for k in range(K):

        @parameter
        for n in range(N):
            b_host[n * K + k] = k + n

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = ctx.enqueue_create_buffer[input_type](M * K)
    var b_device = ctx.enqueue_create_buffer[input_type](K * N)
    var c_device = ctx.enqueue_create_buffer[DType.float32](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[DType.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    alias kernel_func = test_ldmatrix_fp8[input_type]
    ctx.enqueue_function_checked[kernel_func, kernel_func](
        c_device,
        a_device,
        b_device,
        grid_dim=1,
        block_dim=WARP_SIZE,
    )

    ctx.enqueue_copy(c_host, c_device)

    # Run naive matmul.
    alias BLOCK_DIM = 16

    # Create LayoutTensors directly
    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var c_tensor_ref = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        c_device_ref,
        RuntimeLayout[layout].row_major(IndexList[2](M, N)),
    )

    var a_tensor = LayoutTensor[input_type, layout, MutableAnyOrigin](
        a_device,
        RuntimeLayout[layout].row_major(IndexList[2](M, K)),
    )

    var b_tensor = LayoutTensor[input_type, layout, MutableAnyOrigin](
        b_device,
        RuntimeLayout[layout].row_major(
            IndexList[2](N, K)
        ),  # N x K for transpose_b=True
    )

    alias kernel = matmul_kernel_naive[
        DType.float32,
        input_type,
        input_type,
        c_tensor_ref.layout,
        a_tensor.layout,
        b_tensor.layout,
        BLOCK_DIM,
        transpose_b=True,
    ]
    ctx.enqueue_function_checked[kernel, kernel](
        c_tensor_ref,
        a_tensor,
        b_tensor,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    ctx.enqueue_copy(c_host_ref, c_device_ref)

    ctx.synchronize()

    for i in range(M * N):
        var out_val = c_host.load(i)
        var out_ref = c_host_ref.load(i)
        assert_almost_equal(out_val, out_ref)

    _ = a_device
    _ = b_device
    _ = c_device
    _ = c_device_ref

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_ref


def main():
    with DeviceContext() as ctx:
        check_ldmatrix_fp8[DType.float8_e4m3fn](ctx)
        check_ldmatrix_fp8[DType.float8_e5m2](ctx)
