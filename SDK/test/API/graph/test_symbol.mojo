# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo -D TEMP_DIR=%T %s

from tensor import Tensor, TensorShape

from max.graph import testing, Module, MOTensor, Symbol, TypeTuple


fn test_replace_all_uses_with() raises:
    alias name = "test_replace_all_uses_with"

    var m = Module()
    m.m.context().print_on_diagnostic()
    var g = m.graph(
        name,
        TypeTuple(MOTensor(DType.int64), MOTensor(DType.int64)),
        MOTensor(DType.int64),
    )
    g.output(g[0] + g[1])
    m.verify()

    let x = Tensor[DType.int64](TensorShape(), 1)
    let y = Tensor[DType.int64](TensorShape(), 2)
    g[0].replace_all_uses_with(g[1] + 3)
    let expected = Tensor[DType.int64](TensorShape(), 7)

    let actual = testing.execute_binary[DType.int64, DType.int64](m, name, x, y)
    testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_replace_all_uses_with_transform() raises:
    alias name = "test_replace_all_uses_with_transform"

    var m = Module()
    m.m.context().print_on_diagnostic()
    var g = m.graph(
        name,
        TypeTuple(MOTensor(DType.int64), MOTensor(DType.int64)),
        MOTensor(DType.int64),
    )
    g.output(g[0] + g[1])
    m.verify()

    let x = Tensor[DType.int64](TensorShape(), 1)
    let y = Tensor[DType.int64](TensorShape(), 2)

    fn transform(arg0: Symbol) raises -> Symbol:
        return arg0 + 3

    g[0].replace_all_uses_with(transform)
    let expected = Tensor[DType.int64](TensorShape(), 6)

    let actual = testing.execute_binary[DType.int64, DType.int64](m, name, x, y)
    testing.assert_tensors_equal[DType.int64](expected, actual)


def main():
    test_replace_all_uses_with()
    test_replace_all_uses_with_transform()
