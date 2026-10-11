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

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.repeat_interleave import _repeat_interleave_gpu
from std.testing import assert_equal, TestSuite


def test_repeat_interleave_gpu_rank1() raises:
    """Scalar repeats expand each element into consecutive output slots.

    [1, 2, 3] with repeats=2 gives [1, 1, 2, 2, 3, 3]: every input
    element maps to `repeats` consecutive output positions, so the
    source index decodes as output_index // repeats.
    """
    comptime input_layout = row_major[3]()
    comptime output_layout = row_major[6]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.float32](3)
        input_host[0] = Float32(1.0)
        input_host[1] = Float32(2.0)
        input_host[2] = Float32(3.0)
        # The repeats buffer mirrors the graph path: a size-1 int64
        # buffer on the input's device.
        var repeats_host = ctx.enqueue_create_host_buffer[.int64](1)
        repeats_host[0] = 2

        var input_device = ctx.enqueue_create_buffer[.float32](3)
        var output_device = ctx.enqueue_create_buffer[.float32](6)
        var repeats_device = ctx.enqueue_create_buffer[.int64](1)
        ctx.enqueue_copy(input_device, input_host)
        ctx.enqueue_copy(repeats_device, repeats_host)

        _repeat_interleave_gpu[target="gpu"](
            TileTensor(input_device, input_layout),
            TileTensor(repeats_device, row_major(1)),
            0,
            TileTensor(output_device, output_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.float32](6)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        var values = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        for i in range(6):
            assert_equal(output_host[i], Float32(values[i]))


def test_repeat_interleave_gpu_rank2_axis1() raises:
    """Rank-2 input repeats along the last axis; other axes untouched.

    [[1, 2], [3, 4]] with repeats=3, axis=1 gives
    [[1,1,1,2,2,2],[3,3,3,4,4,4]].
    """
    comptime input_layout = row_major[2, 2]()
    comptime output_layout = row_major[2, 6]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.float32](4)
        input_host[0] = Float32(1.0)
        input_host[1] = Float32(2.0)
        input_host[2] = Float32(3.0)
        input_host[3] = Float32(4.0)
        var repeats_host = ctx.enqueue_create_host_buffer[.int64](1)
        repeats_host[0] = 3

        var input_device = ctx.enqueue_create_buffer[.float32](4)
        var output_device = ctx.enqueue_create_buffer[.float32](12)
        var repeats_device = ctx.enqueue_create_buffer[.int64](1)
        ctx.enqueue_copy(input_device, input_host)
        ctx.enqueue_copy(repeats_device, repeats_host)

        _repeat_interleave_gpu[target="gpu"](
            TileTensor(input_device, input_layout),
            TileTensor(repeats_device, row_major(1)),
            1,
            TileTensor(output_device, output_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.float32](12)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        var values = [
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            3.0,
            3.0,
            3.0,
            4.0,
            4.0,
            4.0,
        ]
        for i in range(12):
            assert_equal(output_host[i], Float32(values[i]))


def main() raises:
    var suite = TestSuite()
    suite.test[test_repeat_interleave_gpu_rank1]()
    suite.test[test_repeat_interleave_gpu_rank2_axis1]()
    suite^.run()
