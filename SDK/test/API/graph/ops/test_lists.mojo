# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s
from tensor import Tensor, TensorShape
import testing

from max.graph import _testing, Module, TypeTuple, MOList, MOTensor, SymbolTuple
from max.graph import ops


fn test_list_empty() raises:
    var m = Module()
    var g = m.graph(TypeTuple(), MOList(MOTensor(DType.float32, 1)))
    var l = ops.list(MOTensor(DType.float32, 1), g)
    g.output(l)

    m.verify()
    testing.assert_true("mo.list.create" in str(m))

    var lst = _testing.execute_nullary_list(m)
    testing.assert_equal(len(lst), 0)


fn test_list_nonempty() raises:
    var m = Module()
    var g = m.graph(
        "test_list_nonempty",
        MOTensor(DType.float32, 1),
        MOList(MOTensor(DType.float32, 1)),
    )
    var l = ops.list((g[0]))
    g.output(l)

    var x = Tensor[DType.float32](TensorShape(1), 1.0)

    m.verify()
    testing.assert_true("mo.list.create" in str(m))
    print(m)

    var lst = _testing.execute_unary_list(m, x)
    testing.assert_equal(len(lst), 1)
    _testing.assert_tensors_equal(lst[0], x)


fn test_list_empty_wrong_overload() raises:
    with testing.assert_raises():
        _ = ops.list(SymbolTuple())


fn test_list_mismatched_types() raises:
    var m = Module()
    var g = m.graph(
        "test_list_mismatched_types",
        TypeTuple(MOTensor(DType.float32, 1), MOTensor(DType.float32, 1, 2)),
        TypeTuple(),
    )
    with testing.assert_raises():
        _ = ops.list((g[0], g[1]))


fn main() raises:
    test_list_empty()
    test_list_nonempty()
    test_list_empty_wrong_overload()
    test_list_mismatched_types()
