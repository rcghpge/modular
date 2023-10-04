# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: disabled
# For some reason it crashes in CI
# UNSUPPORTED: windows
# RUN: mojo -I %mobi_pkg_dir %s | FileCheck %s

import mobi


fn test_empty_graph() raises:
    var m = mobi.Module()
    _ = m.new_graph(
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
    var g = m.new_graph(
        "identity_graph",
        m.types(m.si32_scalar()),
        m.types(m.si32_scalar()),
    )
    _ = m.new_op(
        g,
        "mo.output",
        m.values(m.get_graph_input(g, 0)),
        m.void(),
    )

    # CHECK: module {
    # CHECK:   mo.graph @identity_graph([[arg:.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[], si32>
    # CHECK:     mo.output [[arg]] : !mo.tensor<[], si32>
    # CHECK:   }
    # CHECK: }
    _ = m.debug_print()


fn main() raises:
    test_empty_graph()
    test_identity_graph()
