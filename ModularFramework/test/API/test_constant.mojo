# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo -I %mobi_pkg_dir %s

from testing import *

from mobi import *


fn test_basic_i64() raises:
    var m = Module()
    var g = m.graph("test_basic_i64", Arity(), Arity(m.i64(1)))
    let y = g.i64_constant(m.i64(1), 1)
    _ = g.op("mo.output", y, Arity())

    let ir = y.__str__()
    assert_not_equal(ir.find("value = #M.dense_array<1> : tensor<1xsi64>"), -1)
    assert_not_equal(ir.find(": !mo.tensor<[1], si64>"), -1)


fn main() raises:
    test_basic_i64()
