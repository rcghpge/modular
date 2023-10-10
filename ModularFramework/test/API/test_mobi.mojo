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
    _ = g.op("mo.output", g.values(g.input(0)), m.void())

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
    let y = g.op(
        "mo.add", g.values(g.input(0), g.input(1)), m.types(m.i32_scalar())
    )
    _ = g.op("mo.output", y, m.void())

    # CHECK-LABEL: @basic_add
    # CHECK: mo.add
    _ = m.debug_print()


fn test_basic_add_sugary() raises:
    var m = mobi.Module()
    var g = m.graph(
        "basic_add_sugary",
        m.types(m.scalar[DType.int32](), m.scalar[DType.int32]()),
        m.types(m.scalar[DType.int32]()),
    )
    let y = g.op(
        "mo.add",
        g.values(g.inputs[0], g.inputs[1]),
        m.types(m.scalar[DType.int32]()),
    )
    _ = g.op("mo.output", y, m.void())

    # CHECK-LABEL: @basic_add_sugary
    # CHECK: mo.add
    _ = m.debug_print()


fn main() raises:
    test_empty_graph()
    test_identity_graph()
    test_basic_add()
    test_basic_add_sugary()
