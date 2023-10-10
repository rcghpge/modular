# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: address
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
    builder.add_type(ins, intScalar)

    var outs = builder.new_arity()
    builder.add_type(outs, intScalar)

    var unknown = builder.unknown_loc(m)
    var g = builder.new_graph(m, unknown, "identity_graph", ins, outs)

    var rets = builder.new_tuple()
    var arg0 = builder.get_input(g, 0)
    builder.add_symbol(rets, arg0)

    var void = builder.new_arity()

    var attrs = builder.new_attr_map()

    _ = builder.new_op(g, unknown, "mo.output", rets, void, attrs)

    # CHECK: mo.graph @identity_graph
    # CHECK: mo.output
    builder.debug_print_module(m)


fn test_tuple_copy() raises:
    var builder = mobi.Builder()

    var m = builder.new_module()

    var ins = builder.new_arity()
    var intScalar = builder.si32_scalar(m)
    builder.add_type(ins, intScalar)

    var outs = builder.new_arity()
    builder.add_type(outs, intScalar)

    var unknown = builder.unknown_loc(m)
    var g = builder.new_graph(m, unknown, "tuple_copy_test", ins, outs)

    var t = builder.new_tuple()
    var arg0 = builder.get_input(g, 0)
    builder.add_symbol(t, arg0)

    var t2 = builder.new_tuple()
    var el = builder.get_symbol(t, 0)
    builder.add_symbol(t2, el)

    var void = builder.new_arity()

    var attrs = builder.new_attr_map()

    _ = builder.new_op(g, unknown, "mo.output", t2, void, attrs)

    # CHECK: mo.graph @tuple_copy_test
    # CHECK: mo.output
    builder.debug_print_module(m)


fn test_const_op() raises:
    var builder = mobi.Builder()

    var m = builder.new_module()

    var void = builder.new_arity()

    var outs = builder.new_arity()
    var intScalar = builder.si32_scalar(m)
    builder.add_type(outs, intScalar)

    var unknown = builder.unknown_loc(m)
    var g = builder.new_graph(m, unknown, "const_op", void, outs)

    var attrs = builder.new_attr_map()
    var val: Int32 = 1
    var a = builder.tensor_attr(
        m, "value", Pointer[Int32].address_of(val).bitcast[AnyType](), intScalar
    )
    builder.add_attr(attrs, a)

    let noIns = builder.new_tuple()
    var x = builder.new_op(g, unknown, "mo.constant", noIns, outs, attrs)

    var rets = builder.new_tuple()
    var x0 = builder.get_symbol(x, 0)
    builder.add_symbol(rets, x0)

    var noAttrs = builder.new_attr_map()

    _ = builder.new_op(g, unknown, "mo.output", rets, void, noAttrs)

    # CHECK: mo.graph @const_op
    # CHECK: mo.constant
    builder.debug_print_module(m)


fn main() raises:
    test_empty_module()
    test_empty_graph()
    test_identity_graph()
    test_tuple_copy()
    test_const_op()
