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

from time import perf_counter_ns

import gridv1
import gridv2
import gridv3


def main():
    alias warmup_iterations = 10
    alias benchmark_iterations = 1000
    alias rows = 1024
    alias cols = 1024

    # Initial state
    gridv1 = gridv1.Grid.random(rows, cols, seed=42)
    gridv2 = gridv2.Grid[rows, cols].random(seed=42)
    gridv3 = gridv3.Grid[rows, cols].random(seed=42)

    # Warm up
    warmv1 = gridv1
    for _ in range(warmup_iterations):
        warmv1 = warmv1.evolve()

    warmv2 = gridv2
    for _ in range(warmup_iterations):
        warmv2 = warmv2.evolve()

    warmv3 = gridv3
    for _ in range(warmup_iterations):
        warmv3 = warmv3.evolve()

    # Benchmark
    start_time = perf_counter_ns()
    for _ in range(benchmark_iterations):
        gridv1 = gridv1.evolve()
    stop_time = perf_counter_ns()
    elapsed = round((stop_time - start_time) / 1e6, 3)
    print(
        benchmark_iterations,
        "evolutions of gridv1.Grid elapsed time: ",
        elapsed,
        "ms",
    )

    start_time = perf_counter_ns()
    for _ in range(benchmark_iterations):
        gridv2 = gridv2.evolve()
    stop_time = perf_counter_ns()
    elapsed = round((stop_time - start_time) / 1e6, 3)
    print(
        benchmark_iterations,
        "evolutions of gridv2.Grid elapsed time: ",
        elapsed,
        "ms",
    )

    start_time = perf_counter_ns()
    for _ in range(benchmark_iterations):
        gridv3 = gridv3.evolve()
    stop_time = perf_counter_ns()
    elapsed = round((stop_time - start_time) / 1e6, 3)
    print(
        benchmark_iterations,
        "evolutions of gridv3.Grid elapsed time: ",
        elapsed,
        "ms",
    )
