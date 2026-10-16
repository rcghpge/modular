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
"""Benchmark the fused four-stream mHC step."""

from std.sys import get_defined_int

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, row_major
from nn.mega_mhc import MEGA_MHC_COUNTERS, mega_mhc, mega_mhc_scratch_size


def _input[
    dtype: DType
](
    ctx: DeviceContext, count: Int, seed: Int, scale: Float32
) raises -> DeviceBuffer[dtype]:
    var host = ctx.enqueue_create_host_buffer[dtype](count)
    for i in range(count):
        host[i] = (Float32((i * 17 + seed * 13) % 97 - 48) / 64 * scale).cast[
            dtype
        ]()
    var device = ctx.enqueue_create_buffer[dtype](count)
    ctx.enqueue_copy(device, host)
    return device^


def bench_mega_mhc[
    hidden: Int, shifted: Bool, count: Int
](ctx: DeviceContext, mut bench: Bench) raises:
    var x = _input[.bfloat16](ctx, count * hidden, 1, 1)
    var residual = _input[.bfloat16](ctx, count * 4 * hidden, 2, 1)
    var post = _input[.float32](ctx, count * 4, 3, 0.5)
    var comb = _input[.float32](ctx, count * 16, 4, 0.25)
    var previous_pre = _input[.float32](ctx, count * 4, 5, 0.5)
    var projection = _input[.float32](ctx, 24 * 4 * hidden, 6, 0.01)
    var mix_scales = _input[.float32](ctx, 3, 7, 0.1)
    var bias = _input[.float32](ctx, 24, 8, 0.1)
    var weight = _input[.bfloat16](ctx, hidden, 9, 1)
    var new_residual = ctx.enqueue_create_buffer[.bfloat16](count * 4 * hidden)
    var new_pre = ctx.enqueue_create_buffer[.float32](count * 4)
    var new_post = ctx.enqueue_create_buffer[.float32](count * 4)
    var new_comb = ctx.enqueue_create_buffer[.float32](count * 16)
    var output = ctx.enqueue_create_buffer[.bfloat16](count * hidden)
    var counters = ctx.enqueue_create_buffer[.uint64](MEGA_MHC_COUNTERS)
    counters.enqueue_fill(0)
    var scratch_size = mega_mhc_scratch_size[hidden, shifted](count)
    var scratch = ctx.enqueue_create_buffer[.float32](scratch_size)

    var t_new_residual = TileTensor(new_residual, row_major(count * 4 * hidden))
    var t_new_pre = TileTensor(new_pre, row_major(count * 4))
    var t_new_post = TileTensor(new_post, row_major(count * 4))
    var t_new_comb = TileTensor(new_comb, row_major(count * 16))
    var t_output = TileTensor(output, row_major(count * hidden))
    var t_counters = TileTensor(counters, row_major(MEGA_MHC_COUNTERS))
    var t_scratch = TileTensor(scratch, row_major(scratch_size))

    def launch(c: DeviceContext) raises {imm}:
        mega_mhc[hidden, shifted](
            t_new_residual,
            t_new_pre,
            t_new_post,
            t_new_comb,
            t_output,
            TileTensor(x, row_major(count * hidden)),
            TileTensor(residual, row_major(count * 4 * hidden)),
            TileTensor(post, row_major(count * 4)),
            TileTensor(comb, row_major(count * 16)),
            TileTensor(previous_pre, row_major(count * 4)),
            TileTensor(projection, row_major(24 * 4 * hidden)),
            TileTensor(mix_scales, row_major(3)),
            TileTensor(bias, row_major(24)),
            TileTensor(weight, row_major(hidden)),
            t_counters,
            t_scratch,
            2e-5,
            3e-4,
            1.25,
            2e-6,
            10,
            7e-6,
            1.25,
            c,
        )

    def measure(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, launch, ctx)

    # x, residual, new_residual and output once each: the logical traffic.
    var total_bytes = count * hidden * 2 * 10
    bench.bench_function(
        measure,
        BenchId(
            "mega_mhc",
            input_id=String(
                "hidden=", hidden, "/shifted=", shifted, "/tokens=", count
            ),
        ),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )
    ctx.synchronize()


def main() raises:
    comptime hidden = get_defined_int["hidden", 4096]()
    comptime shifted = get_defined_int["shifted", 0]() != 0
    comptime tokens = get_defined_int["tokens", 4096]()

    var bench = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        bench_mega_mhc[hidden, shifted, tokens](ctx, bench)
    bench.dump_report()
