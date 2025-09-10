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

from gpu.host import DeviceContext, FuncAttribute
from gpu.id import thread_idx, global_idx, block_dim
from gpu.memory import AddressSpace, external_memory
from gpu.sync import barrier
from testing import assert_equal, assert_almost_equal
from sys import align_of


fn test_external_shared_mem(ctx: DeviceContext) raises:
    print("== test_external_shared_mem")

    fn dynamic_smem_kernel(data: UnsafePointer[Float32]):
        var dynamic_sram = external_memory[
            Float32, address_space = AddressSpace.SHARED, alignment=4
        ]()
        dynamic_sram[thread_idx.x] = thread_idx.x
        barrier()
        data[thread_idx.x] = dynamic_sram[thread_idx.x]

    var res_host_ptr = UnsafePointer[Float32].alloc(16)
    var res_device = ctx.enqueue_create_buffer[DType.float32](16)

    for i in range(16):
        res_host_ptr[i] = 0

    ctx.enqueue_copy(res_device, res_host_ptr)

    alias kernel = dynamic_smem_kernel
    ctx.enqueue_function_checked[kernel, kernel](
        res_device,
        grid_dim=1,
        block_dim=16,
        shared_mem_bytes=64 * 1024,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(64 * 1024),
    )

    ctx.enqueue_copy(res_host_ptr, res_device)

    ctx.synchronize()

    var expected: List[Float32] = [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
    ]
    for i in range(16):
        print(res_host_ptr[i])
        assert_equal(res_host_ptr[i], expected[i])

    _ = res_device
    res_host_ptr.free()


# Kernel that uses shared memory for testing occupancy with dynamic shared memory
fn shared_memory_kernel(
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    len: Int,
):
    """A kernel that uses shared memory to test occupancy calculations."""
    var tid = global_idx.x
    var thread_id = thread_idx.x
    var block_size = block_dim.x

    # Get a pointer to shared memory for the indices and values
    var shared_data = external_memory[
        Float32,
        address_space = AddressSpace.SHARED,
        alignment = align_of[Float32](),
    ]()

    # Load data into shared memory
    if tid < UInt(len):
        shared_data[thread_id] = input[tid]
    else:
        shared_data[thread_id] = 0.0

    # Synchronize threads in the block
    barrier()

    # Simple processing: add neighboring elements
    var result = shared_data[thread_id]
    if thread_id > 0:
        result += shared_data[thread_id - 1]
    if thread_id < block_size - 1:
        result += shared_data[thread_id + 1]

    # Write result back
    if tid < UInt(len):
        output[tid] = result


# Simple kernel for testing occupancy calculations
fn occupancy_test_kernel(
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    len: Int,
):
    """A simple kernel for testing occupancy - just copies input to output."""
    var tid = global_idx.x
    if tid >= UInt(len):
        return
    output[tid] = input[tid] * 2.0


fn test_occupancy_max_active_blocks(ctx: DeviceContext) raises:
    print(
        "Testing occupancy_max_active_blocks_per_multiprocessor"
        " functionality..."
    )

    # Compile the simple kernel for occupancy testing
    var simple_func = ctx.compile_function[occupancy_test_kernel]()

    # Test with different block sizes
    var block_sizes = List[Int](32, 64, 128, 256, 512, 1024)

    print("\nTesting occupancy with different block sizes (no shared memory):")
    for i in range(len(block_sizes)):
        var block_size = block_sizes[i]
        var max_blocks = (
            simple_func.occupancy_max_active_blocks_per_multiprocessor(
                block_size, 0  # 0 bytes of dynamic shared memory
            )
        )
        print(
            "Block size: ",
            block_size,
            ", Max active blocks per SM: ",
            max_blocks,
        )

        # Basic sanity checks
        assert_equal(max_blocks >= 0, True, "Max blocks should be non-negative")

        # For very large block sizes, we expect fewer blocks per SM
        if block_size >= 512:
            assert_equal(
                max_blocks <= 4, True, "Large blocks should limit occupancy"
            )

    # Test with shared memory usage
    var shared_func = ctx.compile_function[shared_memory_kernel]()

    print("\nTesting occupancy with different shared memory sizes:")
    var shared_memory_sizes = List[Int](
        0, 1024, 4096, 8192, 16384, 32768
    )  # bytes

    for i in range(len(shared_memory_sizes)):
        var shared_mem_size = shared_memory_sizes[i]
        var max_blocks = (
            shared_func.occupancy_max_active_blocks_per_multiprocessor(
                128, shared_mem_size  # Fixed block size of 128 threads
            )
        )
        print(
            "Shared memory: ",
            shared_mem_size,
            "bytes, Max active blocks per SM: ",
            max_blocks,
        )

        # Basic sanity check
        assert_equal(max_blocks >= 0, True, "Max blocks should be non-negative")

    # Test edge cases
    print("\nTesting edge cases:")

    # Test with minimum and maximum block sizes
    var min_blocks = simple_func.occupancy_max_active_blocks_per_multiprocessor(
        1, 0
    )
    var max_thread_blocks = (
        simple_func.occupancy_max_active_blocks_per_multiprocessor(1024, 0)
    )

    print("Min block size (1):", min_blocks, "blocks per SM")
    print("Max block size (1024):", max_thread_blocks, "blocks per SM")

    # Verify that occupancy makes sense
    assert_equal(
        min_blocks >= max_thread_blocks,
        True,
        "Smaller blocks should allow equal or higher occupancy",
    )

    # Test actual kernel execution to verify the function works
    print("\nVerifying kernel execution with optimized block size:")

    alias length = 1024
    var input_host = ctx.enqueue_create_host_buffer[DType.float32](length)
    var output_host = ctx.enqueue_create_host_buffer[DType.float32](length)

    # Initialize input data
    for i in range(length):
        input_host[i] = Float32(i)

    var input_device = ctx.enqueue_create_buffer[DType.float32](length)
    var output_device = ctx.enqueue_create_buffer[DType.float32](length)

    # Copy input to device
    ctx.enqueue_copy(input_device, input_host)

    # Use a block size that should have good occupancy
    var optimal_block_size = 128
    var optimal_blocks = (
        simple_func.occupancy_max_active_blocks_per_multiprocessor(
            optimal_block_size, 0
        )
    )
    print(
        "Using block size ",
        optimal_block_size,
        " with ",
        optimal_blocks,
        " max blocks per SM",
    )

    # Launch the kernel
    var grid_dim = (length + optimal_block_size - 1) // optimal_block_size
    alias kernel = occupancy_test_kernel
    ctx.enqueue_function_checked[kernel, kernel](
        input_device,
        output_device,
        length,
        grid_dim=grid_dim,
        block_dim=optimal_block_size,
    )

    # Copy results back
    ctx.enqueue_copy(output_host, output_device)
    ctx.synchronize()

    # Verify results
    for i in range(min(10, length)):
        var expected = Float32(i) * 2.0
        assert_almost_equal(output_host[i], expected, atol=1e-6)


def main():
    with DeviceContext() as ctx:
        test_external_shared_mem(ctx)
        test_occupancy_max_active_blocks(ctx)
