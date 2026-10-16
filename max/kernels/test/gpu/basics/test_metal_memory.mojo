# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from std.memory import unsafe_stack_allocation
from std.sys.intrinsics import strided_load, strided_store
from std.testing import assert_equal

from max.gpu.host import DeviceContext

comptime _N = 8


def _kernel(output: MutPointer[Float32, MutAnyOrigin]):
    # Threadgroup memory is where losing the address space shows up: a
    # pointer rebuilt from an integer address cannot reach it on Metal.
    var shared = unsafe_stack_allocation[
        _N, DType.float32, address_space=.SHARED
    ]()
    for i in range(_N):
        shared.unsafe_offset(i).unsafe_store(Float32(i))

    var gathered = shared.unsafe_gather(SIMD[.int32, 4](6, 4, 2, 0))
    var strided = strided_load[4](shared, 2)
    shared.unsafe_scatter(
        SIMD[.int32, 4](1, 3, 5, 7), SIMD[.float32, 4](-1, -3, -5, -7)
    )
    strided_store(SIMD[.float32, 4](10, 20, 30, 40), shared, 2)
    for i in range(4):
        output.unsafe_offset(i).unsafe_store(gathered[i])
        output.unsafe_offset(4 + i).unsafe_store(strided[i])
        output.unsafe_offset(8 + i).unsafe_store(
            shared.unsafe_offset(2 * i + 1).unsafe_load()
        )
        output.unsafe_offset(12 + i).unsafe_store(
            shared.unsafe_offset(2 * i).unsafe_load()
        )


def main() raises:
    with DeviceContext() as ctx:
        var out_device = ctx.enqueue_create_buffer[.float32](16)
        ctx.enqueue_function[_kernel](out_device, grid_dim=1, block_dim=1)
        # One row of 4 per operation, in the order `_kernel` writes them.
        var expected: List[SIMD[DType.float32, 4]] = [
            SIMD[DType.float32, 4](6, 4, 2, 0),  # unsafe_gather
            SIMD[DType.float32, 4](0, 2, 4, 6),  # strided_load
            SIMD[DType.float32, 4](-1, -3, -5, -7),  # unsafe_scatter
            SIMD[DType.float32, 4](10, 20, 30, 40),  # strided_store
        ]
        with out_device.map_to_host() as out_host:
            for row in range(len(expected)):
                for i in range(4):
                    assert_equal(
                        out_host[4 * row + i],
                        expected[row][i],
                        msg=String("row ", row, " lane ", i),
                    )
