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

from sys.info import align_of

from gpu.host import DeviceContext, FuncAttribute
from gpu.id import thread_idx
from gpu.memory import AddressSpace, external_memory
from gpu.sync import barrier
from memory import stack_allocation
from testing import assert_equal


def test_external_shared_mem(ctx: DeviceContext):
    fn dynamic_smem_kernel(data: UnsafePointer[Float32]):
        var sram = stack_allocation[
            16,
            Float32,
            address_space = AddressSpace.SHARED,
        ]()
        var dynamic_sram = external_memory[
            Float32,
            address_space = AddressSpace.SHARED,
            alignment = align_of[Float32](),
        ]()
        dynamic_sram[thread_idx.x] = thread_idx.x
        sram[thread_idx.x] = thread_idx.x
        barrier()
        data[thread_idx.x] = dynamic_sram[thread_idx.x] + sram[thread_idx.x]

    var res_host_ptr = UnsafePointer[Float32].alloc(16)
    var res_device = ctx.enqueue_create_buffer[DType.float32](16)

    for i in range(16):
        res_host_ptr[i] = 0

    ctx.enqueue_copy(res_device, res_host_ptr)

    alias kernel_func = dynamic_smem_kernel
    ctx.enqueue_function_checked[kernel_func, kernel_func, dump_llvm=True](
        res_device,
        grid_dim=1,
        block_dim=16,
        shared_mem_bytes=24960,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(24960),
    )

    ctx.enqueue_copy(res_host_ptr, res_device)

    for i in range(16):
        assert_equal(res_host_ptr[i], 2 * i)

    _ = res_device
    res_host_ptr.free()


def main():
    with DeviceContext() as ctx:
        test_external_shared_mem(ctx)
