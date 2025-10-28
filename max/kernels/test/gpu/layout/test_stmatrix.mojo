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
from random import random_si64

from gpu import WARP_SIZE, barrier, lane_id, thread_idx
from gpu.host import DeviceContext
from gpu.mma import ld_matrix, mma, st_matrix
from layout import UNKNOWN_VALUE, Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from layout.tensor_core import get_fragment_size, get_mma_shape
from linalg.matmul.gpu import matmul_kernel_naive
from memory import stack_allocation
from testing import assert_almost_equal

from utils.index import IndexList
from utils.numerics import get_accum_type


fn test_stmatrix(
    c_ptr: UnsafePointer[Float32],
    a_ptr: UnsafePointer[Float32],
    b_ptr: UnsafePointer[Float32],
    m: Int,
    n: Int,
    k: Int,
):
    alias mma_m: UInt = 16
    alias mma_n: UInt = 8
    alias mma_k: UInt = 8

    var d_reg = SIMD[DType.float32, 4](0)
    var tid = thread_idx.x
    var a_shared = stack_allocation[
        mma_m * mma_k,
        DType.float32,
        alignment=32,
        address_space = AddressSpace.SHARED,
    ]()
    var b_shared = stack_allocation[
        mma_n * mma_k,
        DType.float32,
        alignment=32,
        address_space = AddressSpace.SHARED,
    ]()

    var c_shared = stack_allocation[
        mma_m * mma_n,
        DType.float32,
        alignment=32,
        address_space = AddressSpace.SHARED,
    ]()

    for i in range(tid, mma_m * mma_k, WARP_SIZE):
        a_shared[i] = a_ptr[i]

    # Transpose B to fit ld_matrix layout.
    for i in range(tid, mma_k * mma_n, WARP_SIZE):
        var x = i % mma_n
        var y = i // mma_n
        b_shared[x * mma_k + y] = b_ptr[i]

    barrier()

    var lane = lane_id()
    var a_reg = ld_matrix[4](
        a_shared
        + Int((lane % UInt(m)) * UInt(k) + (lane // UInt(m)) * UInt(k) // 2)
    )
    var b_reg = ld_matrix[2](
        b_shared
        + Int((lane % UInt(k)) * UInt(n) + (lane // UInt(k)) * UInt(n) // 2)
    )

    mma(d_reg, a_reg, b_reg, d_reg)
    st_matrix[4](
        c_shared.offset(thread_idx.x * 4), rebind[SIMD[DType.float32, 4]](d_reg)
    )

    var grp = lane_id() // 16
    var local = lane_id() % 16

    var base = tid * 4
    for i in range(4):
        var d = base + UInt(i)
        var r = d & 63
        var src = ((d >> 6) << 6) + ((r & 1) << 5) + (r >> 1)
        c_ptr[d] = c_shared[src]


fn test_stmatrix_gen[
    input_type: DType, output_type: DType
](
    c_ptr: UnsafePointer[Scalar[output_type]],
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

    var lane = lane_id()
    var d_reg = SIMD[accum_type, c_frag_size](0)

    var a_shared = stack_allocation[
        M * K, input_type, alignment=32, address_space = AddressSpace.SHARED
    ]()
    var b_shared = stack_allocation[
        N * K, input_type, alignment=32, address_space = AddressSpace.SHARED
    ]()

    var c_shared = stack_allocation[
        M * N,
        accum_type,
        alignment=32,
        address_space = AddressSpace.SHARED,
    ]()

    for i in range(lane, M * K, WARP_SIZE):
        a_shared[i] = a_ptr[i]

    # Transpose B to fit ld_matrix layout.
    for i in range(lane, N * K, WARP_SIZE):
        b_shared[i] = b_ptr[i]

    barrier()

    var a_reg = ld_matrix[a_frag_size](
        a_shared
        + Int((lane % UInt(M)) * UInt(K) + (lane // UInt(M)) * UInt(K) // 2)
    )
    var b_reg = ld_matrix[b_frag_size, transpose=True](
        b_shared
        + Int((lane % UInt(K)) * UInt(N) + (lane // UInt(K)) * UInt(N) // 2)
    )

    mma(d_reg, a_reg, b_reg, d_reg)
    st_matrix[c_frag_size](
        c_shared.offset(thread_idx.x * 4),
        rebind[SIMD[DType.float32, c_frag_size]](d_reg),
    )
    var grp = lane_id() // 16
    var local = lane_id() % 16

    var base = thread_idx.x * 4
    for i in range(4):
        var d = base + UInt(i)
        var r = d & 63
        var src = ((d >> 6) << 6) + ((r & 1) << 5) + (r >> 1)
        c_ptr[d] = c_shared[src].cast[output_type]()


fn check_stmatrix_gen[
    input_type: DType,
    output_type: DType,
](ctx: DeviceContext) raises:
    print("== test stmatrix bf16")

    # Shape for a single mma.
    alias accum_type = get_accum_type[input_type]()
    alias mma_shape = get_mma_shape[input_type, accum_type]()
    alias M = mma_shape[0]
    alias N = mma_shape[1]
    alias K = mma_shape[2]

    var a_host = UnsafePointer[Scalar[input_type]].alloc(M * K)
    var b_host = UnsafePointer[Scalar[input_type]].alloc(K * N)
    var c_host = UnsafePointer[Scalar[output_type]].alloc(M * N)
    var c_host_ref = UnsafePointer[Scalar[output_type]].alloc(M * N)

    for i in range(M * K):
        a_host[i] = Scalar[input_type](i)

    for i in range(K * N):
        b_host[i] = Scalar[input_type](i)

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = ctx.enqueue_create_buffer[input_type](M * K)
    var b_device = ctx.enqueue_create_buffer[input_type](K * N)
    var c_device = ctx.enqueue_create_buffer[output_type](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[output_type](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    alias kernel_type = test_stmatrix_gen[input_type, output_type]
    ctx.enqueue_function_checked[kernel_type, kernel_type](
        c_device,
        a_device,
        b_device,
        grid_dim=1,
        block_dim=WARP_SIZE,
    )

    ctx.enqueue_copy(c_host, c_device)

    var c_tensor_ref = LayoutTensor[output_type, Layout.row_major(M, N)](
        c_device_ref
    )
    var a_tensor = LayoutTensor[input_type, Layout.row_major(M, K)](a_device)
    var b_tensor = LayoutTensor[input_type, Layout.row_major(K, N)](b_device)

    # Run naive matmul.
    alias BLOCK_DIM = 16
    alias kernel_naive_type = matmul_kernel_naive[
        output_type,
        input_type,
        input_type,
        c_tensor_ref.layout,
        a_tensor.layout,
        b_tensor.layout,
        BLOCK_DIM,
    ]
    ctx.enqueue_function_checked[kernel_naive_type, kernel_naive_type](
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


fn check_stmatrix(
    M: Int, N: Int, K: Int, rand_min: Int64, rand_max: Int64, ctx: DeviceContext
) raises:
    print("== test stmatrix instruction")

    var a_host = UnsafePointer[Float32].alloc(M * K)
    var b_host = UnsafePointer[Float32].alloc(K * N)
    var c_host = UnsafePointer[Float32].alloc(M * N)
    var c_host_ref = UnsafePointer[Float32].alloc(M * N)

    for i in range(M * K):
        var val = random_si64(rand_min, rand_max)
        a_host[i] = val.cast[DType.float32]()

    for i in range(K * N):
        var val = random_si64(rand_min, rand_max)
        b_host[i] = val.cast[DType.float32]()

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b_device = ctx.enqueue_create_buffer[DType.float32](K * N)
    var c_device = ctx.enqueue_create_buffer[DType.float32](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[DType.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    alias WARP_PER_BLOCK = 1
    alias MMA_M = 16
    alias MMA_N = 8
    alias MMA_K = 8

    ctx.enqueue_function_checked[test_stmatrix, test_stmatrix](
        c_device,
        a_device,
        b_device,
        M,
        N,
        K,
        grid_dim=1,
        block_dim=WARP_SIZE,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host, c_device)

    # Run naive matmul.
    alias BLOCK_DIM = 16

    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var c_tensor_ref = LayoutTensor[DType.float32, layout](
        c_device_ref,
        RuntimeLayout[layout].row_major(IndexList[2](M, N)),
    )

    var a_tensor = LayoutTensor[DType.float32, layout](
        a_device,
        RuntimeLayout[layout].row_major(IndexList[2](M, K)),
    )

    var b_tensor = LayoutTensor[DType.float32, layout](
        b_device,
        RuntimeLayout[layout].row_major(IndexList[2](K, N)),
    )

    alias kernel = matmul_kernel_naive[
        DType.float32,
        DType.float32,
        DType.float32,
        c_tensor_ref.layout,
        a_tensor.layout,
        b_tensor.layout,
        BLOCK_DIM,
    ]
    ctx.enqueue_function_checked[kernel, kernel](
        c_tensor_ref,
        a_tensor,
        b_tensor,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    ctx.synchronize()
    ctx.enqueue_copy(c_host_ref, c_device_ref)

    for i in range(M * N):
        var out_val = c_host[i]
        var out_ref = c_host_ref[i]
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
        check_stmatrix(16, 8, 8, -100, 100, ctx)
        check_stmatrix_gen[DType.bfloat16, DType.bfloat16](ctx)
