# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo -I %mobi_pkg_dir %s | FileCheck %s

import mobi
from tensor import Tensor
from utils.index import Index


fn test_empty_graph() raises:
    var m = mobi.Module()
    _ = m.graph("empty_graph", m.types(), m.types())

    # CHECK: builtin.module
    # CHECK: mo.graph
    # CHECK: sym_name = "empty_graph"
    print(m.__str__())


fn test_identity_graph() raises:
    var m = mobi.Module()
    var g = m.graph(
        name="identity_graph",
        in_types=m.types(m.i32()),
        out_types=m.types(m.i32()),
    )
    _ = g.op(name="mo.output", inputs=g.values(g[0]), out_types=m.types())

    # CHECK: module {
    # CHECK:   mo.graph @identity_graph([[arg:.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[], si32>
    # CHECK:     mo.output [[arg]] : !mo.tensor<[], si32>
    # CHECK:   }
    # CHECK: }
    print(m.__str__())


fn test_basic_add() raises:
    var m = mobi.Module()
    var g = m.graph("basic_add", m.types(m.i32(), m.i32()), m.types(m.i32()))
    let y = g.op("mo.add", g.values(g[0], g[1]), m.types(m.i32()))
    _ = g.op("mo.output", y, m.types())

    # CHECK-LABEL: @basic_add
    # CHECK: mo.add
    print(m.__str__())


fn test_add_constant() raises:
    var m = mobi.Module()
    var g = m.graph("add_constant", m.types(m.f32(1)), m.types(m.f32(1)))
    let x = g.op(
        "mo.constant",
        g.values(),
        m.types(m.f32(1)),
        m.attrs(m.f32_attr("value", m.f32(1), 1.0)),
    )
    let y = g.op("mo.add", g.values(g[0], x[0]), m.types(m.f32(1)))
    _ = g.op("mo.output", y, m.types())

    # CHECK-LABEL: @add_constant
    # CHECK: mo.constant {value = #M.dense_array<1.000000e+00> : tensor<1xf32>}
    # CHECK: mo.add
    print(m.__str__())

    # CHECK{LITERAL}: tensor([[2.0000]], dtype=f32, shape=[1])
    var testInput = Tensor[DType.float32](1)
    testInput[Index(0)] = 1
    m.debug_print_execute(testInput)


fn main() raises:
    test_empty_graph()
    test_identity_graph()
    test_basic_add()
    test_add_constant()
