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


fn test_empty_module() raises:
    var builder = mobi.Builder()

    var m = builder.new_module()
    # CHECK: module {
    builder.debug_print_module(m)


fn test_empty_graph() raises:
    var builder = mobi.Builder()

    var m = builder.new_module()

    var ins = builder.new_arity()

    var outs = builder.new_arity()

    var unknown = builder.unknown_loc(m)
    _ = builder.new_graph(m, unknown, "empty_graph", ins, outs)

    # CHECK: builtin.module
    # CHECK: mo.graph
    # CHECK: sym_name = "empty_graph"
    builder.debug_print_module(m)


fn test_identity_graph() raises:
    var builder = mobi.Builder()

    var m = builder.new_module()

    var ins = builder.new_arity()
    var intScalar = builder.si32_scalar(m)
    builder.inc_arity(ins, intScalar)

    var outs = builder.new_arity()
    builder.inc_arity(outs, intScalar)

    var unknown = builder.unknown_loc(m)
    var g = builder.new_graph(m, unknown, "identity_graph", ins, outs)

    var rets = builder.new_tuple()
    var arg0 = builder.get_graph_input(g, 0)
    builder.ext_tuple(rets, arg0)

    var void = builder.new_arity()

    _ = builder.new_op(g, unknown, "mo.output", rets, void)

    # CHECK: mo.graph @identity_graph
    # CHECK: mo.output
    builder.debug_print_module(m)


fn main() raises:
    test_empty_module()
    test_empty_graph()
    test_identity_graph()
