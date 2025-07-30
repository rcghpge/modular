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
from gpu.id import block_dim, block_idx, thread_idx
from math import iota
from sys import exit
from sys.info import has_accelerator

alias num_elements = 20


fn scalar_add(vector: UnsafePointer[Float32], size: Int, scalar: Float32):
    """
    Kernel function to add a scalar to all elements of a vector.

    This kernel function adds a scalar value to each element of a vector stored
    in GPU memory. The input vector is modified in place.

    Args:
        vector: Pointer to the input vector.
        size: Number of elements in the vector.
        scalar: Scalar to add to the vector.

    """

    # Calculate the global thread index within the entire grid. Each thread
    # processes one element of the vector.
    #
    # block_idx.x: index of the current thread block.
    # block_dim.x: number of threads per block.
    # thread_idx.x: index of the current thread within its block.
    idx = block_idx.x * block_dim.x + thread_idx.x

    # Bounds checking: ensure we don't access memory beyond the vector size.
    # This is crucial when the number of threads doesn't exactly match vector
    # size.
    if idx < size:
        # Each thread adds the scalar to its corresponding vector element
        # This operation happens in parallel across all GPU threads
        vector[idx] += scalar


def main():
    @parameter
    if not has_accelerator():
        print("No GPUs detected")
        exit(0)
    else:
        # Initialize GPU context for device 0 (default GPU device).
        ctx = DeviceContext()

        # Create a buffer in host (CPU) memory to store our input data
        host_buffer = ctx.enqueue_create_host_buffer[DType.float32](
            num_elements
        )

        # Wait for buffer creation to complete.
        ctx.synchronize()

        # Fill the host buffer with sequential numbers (0, 1, 2, ..., size-1).
        iota(host_buffer.unsafe_ptr(), num_elements)
        print("Original host buffer:", host_buffer)

        # Create a buffer in device (GPU) memory to store data for computation.
        device_buffer = ctx.enqueue_create_buffer[DType.float32](num_elements)

        # Copy data from host memory to device memory for GPU processing.
        ctx.enqueue_copy(src_buf=host_buffer, dst_buf=device_buffer)

        # Compile the scalar_add kernel function for execution on the GPU.
        scalar_add_kernel = ctx.compile_function_checked[
            scalar_add, scalar_add
        ]()

        # Launch the GPU kernel with the following arguments:
        #
        # - device_buffer: GPU memory containing our vector data
        # - num_elements: number of elements in the vector
        # - Float32(20.0): the scalar value to add to each element
        # - grid_dim=1: use 1 thread block
        # - block_dim=num_elements: use 'num_elements' threads per block (one
        #   thread per vector element)
        ctx.enqueue_function_checked(
            scalar_add_kernel,
            device_buffer,
            num_elements,
            Float32(20.0),
            grid_dim=1,
            block_dim=num_elements,
        )

        # Copy the computed results back from device memory to host memory.
        ctx.enqueue_copy(src_buf=device_buffer, dst_buf=host_buffer)

        # Wait for all GPU operations to complete.
        ctx.synchronize()

        # Display the final results after GPU computation.
        print("Modified host buffer:", host_buffer)
