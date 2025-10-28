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

from asyncrt_test_utils import create_test_device_context, expect_eq
from gpu import *
from gpu.host import DeviceContext
from testing import TestSuite


fn vec_func(
    in0: UnsafePointer[Float32],
    in1: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    len: Int,
):
    var tid = global_idx.x
    if tid >= UInt(len):
        return
    output[tid] = in0[tid] + in1[tid]


def test_multi_function():
    var ctx1 = create_test_device_context()
    var ctx2 = create_test_device_context()
    _run_test_multi_function(ctx1, ctx2)


fn _run_test_multi_function(ctx1: DeviceContext, ctx2: DeviceContext) raises:
    alias length = 1024

    var in0_dev1 = ctx1.enqueue_create_buffer[DType.float32](length)
    var in0_dev2 = ctx2.enqueue_create_buffer[DType.float32](length)
    var in1_dev1 = ctx1.enqueue_create_buffer[DType.float32](length)
    var in1_dev2 = ctx2.enqueue_create_buffer[DType.float32](length)
    var out_dev1 = ctx1.enqueue_create_buffer[DType.float32](length)
    var out_dev2 = ctx2.enqueue_create_buffer[DType.float32](length)

    # Initialize the input and outputs with known values.
    with in0_dev1.map_to_host() as in0_host1, in0_dev2.map_to_host() as in0_host2:
        for i in range(length):
            in0_host1[i] = i
            in0_host2[i] = i

    # Setup right side constants.
    _ = in1_dev1.enqueue_fill(2.0)
    _ = in1_dev2.enqueue_fill(2.0)

    # Write known bad values to out_dev.
    _ = out_dev1.enqueue_fill(101.0)
    _ = out_dev2.enqueue_fill(102.0)

    var block_dim = 32

    ctx1.enqueue_function_experimental[vec_func](
        in0_dev1,
        in1_dev1,
        out_dev1,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    ctx2.enqueue_function_experimental[vec_func](
        in0_dev2,
        in1_dev2,
        out_dev2,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    with out_dev1.map_to_host() as out_host1, out_dev2.map_to_host() as out_host2:
        for i in range(length):
            if i < 10:
                print("at index", i, "the value is", out_host1[i])
                print("at index", i, "the value is", out_host2[i])
            expect_eq(
                out_host1[i],
                i + 2,
                "at index ",
                i,
                " the value is ",
                out_host1[i],
            )
            expect_eq(
                out_host2[i],
                i + 2,
                "at index ",
                i,
                " the value is ",
                out_host2[i],
            )


def main():
    # TODO(MOCO-2556): Use automatic discovery when it can handle global_idx.
    # TestSuite.discover_tests[__functions_in_module()]().run()
    var suite = TestSuite()

    suite.test[test_multi_function]()

    suite^.run()
