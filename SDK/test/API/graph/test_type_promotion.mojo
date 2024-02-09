# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo --debug-level full %s

import benchmark
from testing import assert_equal, assert_raises

from max.graph.type_promotion import promote


fn test_type_promotion() raises:
    assert_equal(promote(DType.int8, DType.int32), DType.int32)
    assert_equal(promote(DType.int8, DType.uint8), DType.int16)
    assert_equal(promote(DType.int64, DType.int64), DType.int64)
    assert_equal(promote(DType.int64, DType.uint64), DType.float16)
    with assert_raises(contains="No legal promotion"):
        _ = promote(DType.invalid, DType.int32)


fn benchmark_promote():
    # Since we aren't yet caching these I wanted to see what a promotion check costs.
    # Re-enable this benchmark to try again. I chose a common DType pair that are also
    # close to the bottom of the lattice, and therefore the most expensive to compute.
    #
    # Results on my M2 Mac:
    # ---------------------
    # Benchmark Report (s)
    # ---------------------
    # Mean: 1.6788825928968216e-05
    # Total: 7.2353459999999998
    # Iters: 430962
    # Warmup Mean: 1.7499999999999998e-05
    # Warmup Total: 3.4999999999999997e-05
    # Warmup Iters: 2
    # Fastest Mean: 1.6765553103659512e-05
    # Slowest Mean: 1.6813796944067385e-05
    #
    # While it could certainly be faster, even large model architectures
    # currently only run this a few hundred times.

    fn bench():
        try:
            _ = promote(DType.bool, DType.int32)
        except:
            pass

    let report = benchmark.run[bench]()
    report.print()


def main():
    test_type_promotion()
    # benchmark_promote()
