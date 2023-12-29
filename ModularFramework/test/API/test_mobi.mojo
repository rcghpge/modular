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


fn test_identity_graph() raises:
    var m = mobi.Module()
    var g = m.graph(
        name="identity_graph",
        in_types=m.types(m.i32()),
        out_types=m.types(m.i32()),
    )
    _ = g.op(name="mo.output", inputs=g[0], out_types=m.types())

    m.verify()
    # CHECK: module {
    # CHECK:   mo.graph @identity_graph([[arg:.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[], si32>
    # CHECK:     mo.output [[arg]] : !mo.tensor<[], si32>
    # CHECK:   }
    # CHECK: }
    print(m.__str__())


fn test_basic_add() raises:
    var m = mobi.Module()
    var g = m.graph("basic_add", m.types(m.i32(), m.i32()), m.types(m.i32()))
    let y = g.op("mo.add", (g[0], g[1]), m.types(m.i32()))
    _ = g.op("mo.output", y, m.types())

    m.verify()
    # CHECK-LABEL: @basic_add
    # CHECK: mo.add
    print(m.__str__())


fn test_add_constant() raises:
    var m = mobi.Module()
    var g = m.graph("add_constant", m.types(m.f32(1)), m.types(m.f32(1)))
    let x = g.op(
        "mo.constant",
        m.types(m.f32(1)),
        m.attrs(m.f32_attr("value", m.f32(1), 1.0)),
    )
    let y = g.op("mo.add", (g[0], x[0]), m.types(m.f32(1)))
    _ = g.op("mo.output", y, m.types())

    m.verify()
    # CHECK-LABEL: @add_constant
    # CHECK: mo.constant {value = #M.dense_array<1.000000e+00> : tensor<1xf32>}
    # CHECK: mo.add
    print(m.__str__())


fn test_int_constant_helper() raises:
    var m = mobi.Module()
    var g = m.graph("int_constant_helper", m.types(), m.types(m.i64(2)))
    let x = g.i64_constant(m.i64(2), 1, 2)
    _ = g.op("mo.output", x, m.types())

    m.verify()
    # CHECK-LABEL: @int_constant_helper
    # CHECK: mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>}
    print(m.__str__())


fn main() raises:
    test_identity_graph()
    test_basic_add()
    test_add_constant()
    test_int_constant_helper()
