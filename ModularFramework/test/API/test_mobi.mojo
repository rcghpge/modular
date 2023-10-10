# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: address
# UNSUPPORTED: windows
# RUN: mojo -I %mobi_pkg_dir %s | FileCheck %s

import mobi


fn test_empty_graph() raises:
    var m = mobi.Module()
    _ = m.graph(
        "empty_graph",
        m.void(),
        m.void(),
    )

    # CHECK: builtin.module
    # CHECK: mo.graph
    # CHECK: sym_name = "empty_graph"
    _ = m.debug_print()


fn test_identity_graph() raises:
    var m = mobi.Module()
    var g = m.graph(
        "identity_graph",
        m.types(m.i32_scalar()),
        m.types(m.i32_scalar()),
    )
    _ = g.op("mo.output", g.values(g[0]), m.void())

    # CHECK: module {
    # CHECK:   mo.graph @identity_graph([[arg:.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[], si32>
    # CHECK:     mo.output [[arg]] : !mo.tensor<[], si32>
    # CHECK:   }
    # CHECK: }
    _ = m.debug_print()


fn test_basic_add() raises:
    var m = mobi.Module()
    var g = m.graph(
        "basic_add",
        m.types(m.i32_scalar(), m.i32_scalar()),
        m.types(m.i32_scalar()),
    )
    let y = g.op("mo.add", g.values(g[0], g[1]), m.types(m.i32_scalar()))
    _ = g.op("mo.output", y, m.void())

    # CHECK-LABEL: @basic_add
    # CHECK: mo.add
    _ = m.debug_print()


fn test_add_constant() raises:
    var m = mobi.Module()
    var g = m.graph(
        "add_constant",
        m.types(m.i32_scalar()),
        m.types(m.i32_scalar()),
    )
    var x = g.op(
        "mo.constant",
        g.no_value(),
        m.types(m.i32_scalar()),
        m.attrs(m.i32_scalar_attr("value", 1)),
    )
    let y = g.op("mo.add", g.values(g[0], x[0]), m.types(m.i32_scalar()))
    _ = g.op("mo.output", y, m.void())

    # CHECK-LABEL: @add_constant
    # CHECK: mo.constant {value = #M.dense_array<1> : tensor<si32>}
    # CHECK: mo.add
    _ = m.debug_print()


fn main() raises:
    test_empty_graph()
    test_identity_graph()
    test_basic_add()
    test_add_constant()
