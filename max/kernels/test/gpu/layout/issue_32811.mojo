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

from gpu.host import DeviceContext
from gpu import block_idx, thread_idx
from layout import *
from layout.layout_tensor import LayoutTensor


fn gpu_kernel(
    dst: UnsafePointer[Float32],
    rhs: UnsafePointer[Float32],
    lhs: UnsafePointer[Float32],
):
    dst[block_idx.x * 4 + thread_idx.x] = (
        rhs[block_idx.x * 4 + thread_idx.x]
        + lhs[block_idx.x * 4 + thread_idx.x]
    )

    _ = LayoutTensor[DType.float32, Layout(IntTuple(16, 1), IntTuple(1, 1))](
        dst
    )


def main():
    with DeviceContext() as ctx:
        var vec_a_ptr = UnsafePointer[Float32].alloc(16)
        var vec_b_ptr = UnsafePointer[Float32].alloc(16)
        var vec_c_ptr = UnsafePointer[Float32].alloc(16)

        var vec_a_dev = ctx.enqueue_create_buffer[DType.float32](16)
        var vec_b_dev = ctx.enqueue_create_buffer[DType.float32](16)
        var vec_c_dev = ctx.enqueue_create_buffer[DType.float32](16)

        for i in range(16):
            vec_a_ptr[i] = i
            vec_b_ptr[i] = i
            vec_c_ptr[i] = 0

        ctx.enqueue_copy(vec_a_dev, vec_a_ptr)
        ctx.enqueue_copy(vec_b_dev, vec_b_ptr)
        ctx.enqueue_copy(vec_c_dev, vec_c_ptr)

        ctx.enqueue_function_checked[gpu_kernel, gpu_kernel](
            vec_c_dev,
            vec_a_dev,
            vec_b_dev,
            block_dim=(4),
            grid_dim=(4),
        )

        ctx.enqueue_copy(vec_a_ptr, vec_a_dev)
        ctx.enqueue_copy(vec_b_ptr, vec_b_dev)
        ctx.enqueue_copy(vec_c_ptr, vec_c_dev)

        ctx.synchronize()

        for i in range(16):
            print(vec_a_ptr[i], "+", vec_b_ptr[i], "=", vec_c_ptr[i])
