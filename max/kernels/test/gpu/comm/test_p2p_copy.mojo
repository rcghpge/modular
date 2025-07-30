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
from sys import env_get_int

from gpu import block_dim, global_idx, grid_dim
from gpu.host import DeviceBuffer, DeviceContext
from testing import assert_almost_equal, assert_true


fn p2p_copy_kernel(
    dst: UnsafePointer[Float32],
    src: UnsafePointer[Float32],
    num_elements: Int,
):
    var tid = global_idx.x
    if tid < num_elements:
        dst[tid] = src[tid]


fn launch_p2p_copy_kernel(
    ctx1: DeviceContext,
    dst_buf: DeviceBuffer[DType.float32],
    src_buf: DeviceBuffer[DType.float32],
    num_elements: Int,
) raises:
    alias BLOCK_SIZE = 256
    var grid_size = ceildiv(num_elements, BLOCK_SIZE)

    # Launch the kernel on both devices
    ctx1.enqueue_function[p2p_copy_kernel](
        dst_buf.unsafe_ptr(),
        src_buf.unsafe_ptr(),
        num_elements,
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
    )

    # Synchronize both contexts to ensure completion
    ctx1.synchronize()


def main():
    alias log2_length = env_get_int["log2_length", 20]()
    constrained[log2_length > 0]()
    var length = 1 << log2_length

    var num_devices = DeviceContext.number_of_devices()
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )

    # Create contexts for both devices
    var ctx1 = DeviceContext(device_id=0)
    var ctx2 = DeviceContext(device_id=1)
    var can_access_p2p = ctx1.can_access(ctx2)
    print("ctx1 can access ctx2: ", can_access_p2p)
    if not can_access_p2p:
        print("Skipping test as ctx1 cannot access ctx2")
        return
    ctx1.enable_peer_access(ctx2)
    print("Checkpoint - successfully enabled peer access")

    # Create and initialize device buffers
    var dst_buf = ctx1.create_buffer_sync[DType.float32](length).enqueue_fill(
        1.0
    )
    var src_buf = ctx2.create_buffer_sync[DType.float32](length)

    # Initialize source data
    with src_buf.map_to_host() as host_data:
        for i in range(length):
            host_data[i] = Float32(i * 0.5)

    # Launch the P2P copy kernel
    launch_p2p_copy_kernel(ctx1, dst_buf, src_buf, length)

    # Wait for the copy to complete
    ctx1.synchronize()

    # Verify the data was copied correctly
    with dst_buf.map_to_host() as host_data:
        for i in range(length):
            assert_almost_equal(host_data[i], Float32(i * 0.5))

    print("P2P Direct Addressing Copy Test Passed")
