# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo -I %mobi_pkg_dir %s

from testing import *
from tensor import TensorShape

from mobi import *


fn test_basic_tensor() raises:
    var m = Module()
    var g = m.graph("test_basic_tensor", Arity(), Arity(m.i32(1, 2)))
    let y = g.constant(Tensor[DType.int32](TensorShape(1, 2), 1, 2))
    g.output(y)

    let ir = y.__str__()
    assert_not_equal(
        ir.find("value = #M.dense_array<1, 2> : tensor<1x2xsi32>"), -1
    )
    assert_not_equal(ir.find(": !mo.tensor<[1, 2], si32>"), -1)


fn test_scalar_basic() raises:
    var m = Module()
    var g = m.graph("test_scalar_basic", Arity(), Arity(m.bool()))
    let y = g.scalar[DType.bool](True)
    g.output(y)

    let ir = y.__str__()
    print(ir)
    assert_not_equal(ir.find("value = #M.dense_array<true> : tensor<i1>"), -1)
    assert_not_equal(ir.find(": !mo.tensor<[], bool>"), -1)


fn test_scalar_high_rank() raises:
    var m = Module()
    var g = m.graph("test_scalar_high_rank", Arity(), Arity(m.bool(1, 1, 1)))
    let y = g.scalar[DType.bool](True, 3)
    g.output(y)

    let ir = y.__str__()
    print(ir)
    assert_not_equal(
        ir.find("value = #M.dense_array<true> : tensor<1x1x1xi1>"), -1
    )
    assert_not_equal(ir.find(": !mo.tensor<[1, 1, 1], bool>"), -1)


fn test_basic_i64() raises:
    var m = Module()
    var g = m.graph("test_basic_i64", Arity(), Arity(m.i64(1)))
    let y = g.scalar(Int64(1), rank=1)
    _ = g.nvop("mo.output", y)

    let ir = y.__str__()
    assert_not_equal(ir.find("value = #M.dense_array<1> : tensor<1xsi64>"), -1)
    assert_not_equal(ir.find(": !mo.tensor<[1], si64>"), -1)


fn main() raises:
    test_basic_tensor()
    test_scalar_basic()
    test_scalar_high_rank()
    test_basic_i64()
