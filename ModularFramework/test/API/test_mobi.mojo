# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo -I %mobi_pkg_dir %s | FileCheck %s

from tensor import Tensor, TensorShape
from utils.index import Index

from mobi import *


fn test_identity_graph() raises:
    var m = Module()
    var g = m.graph(
        name="identity_graph",
        in_types=Arity(m.i32()),
        out_types=Arity(m.i32()),
    )
    _ = g.nvop(name="mo.output", inputs=g[0])

    m.verify()
    # CHECK: module {
    # CHECK:   mo.graph @identity_graph([[arg:.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[], si32>
    # CHECK:     mo.output [[arg]] : !mo.tensor<[], si32>
    # CHECK:   }
    # CHECK: }
    print(m.__str__())


fn test_basic_add() raises:
    var m = Module()
    var g = m.graph("basic_add", Arity(m.i32(), m.i32()), Arity(m.i32()))
    let y = g.op("mo.add", (g[0], g[1]), MOTensor(DType.int32))
    _ = g.nvop("mo.output", y)

    m.verify()
    # CHECK-LABEL: @basic_add
    # CHECK: mo.add
    print(m.__str__())


fn test_add_constant() raises:
    var m = Module()
    var g = m.graph("add_constant", Arity(m.f32(1)), Arity(m.f32(1)))
    let x = g.op(
        "mo.constant",
        MOTensor(DType.float32, 1),
        m.attrs(m.f32_attr("value", m.f32(1), 1.0)),
    )
    let y = g.op("mo.add", (g[0], x), MOTensor(DType.float32, 1))
    _ = g.nvop("mo.output", y)

    m.verify()
    # CHECK-LABEL: @add_constant
    # CHECK: mo.constant {value = #M.dense_array<1.000000e+00> : tensor<1xf32>}
    # CHECK: mo.add
    print(m.__str__())


fn main() raises:
    test_identity_graph()
    test_basic_add()
    test_add_constant()
