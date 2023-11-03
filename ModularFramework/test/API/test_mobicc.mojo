# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: address
# UNSUPPORTED: windows
# RUN: mojo -I %mobi_pkg_dir %s | FileCheck %s

import mobi


fn test_types() raises:
    let builder = mobi.Builder()
    let m = builder.new_module()
    let scalarShape = DynamicVector[Int64]()
    let intScalar = builder.si(m, 32, scalarShape.data, len(scalarShape))

    # CHECK: True
    print(builder.type_is_tensor(intScalar))
    # CHECK: True
    print(builder.type_is_si_tensor(intScalar))
    # CHECK: False
    print(builder.type_is_ui_tensor(intScalar))
    # CHECK: False
    print(builder.type_is_bool_tensor(intScalar))
    # CHECK: 32
    print(builder.tensor_dtype_width(intScalar))
    # CHECK: 0
    print(builder.tensor_rank(intScalar))

    var boolShape = DynamicVector[Int64]()
    boolShape.push_back(2)
    boolShape.push_back(3)
    let boolTensor = builder.bool(m, boolShape.data, len(boolShape))

    # CHECK: True
    print(builder.type_is_tensor(boolTensor))
    # CHECK: False
    print(builder.type_is_si_tensor(boolTensor))
    # CHECK: False
    print(builder.type_is_ui_tensor(boolTensor))
    # CHECK: True
    print(builder.type_is_bool_tensor(boolTensor))
    # CHECK: 8
    print(builder.tensor_dtype_width(boolTensor))
    # CHECK: 2
    print(builder.tensor_rank(boolTensor))
    # CHECK: 2
    print(builder.tensor_dim(boolTensor, 0))
    # CHECK: 3
    print(builder.tensor_dim(boolTensor, 1))

    scalarShape._del_old()
    boolShape._del_old()


fn test_empty_module() raises:
    let builder = mobi.Builder()

    let m = builder.new_module()
    # CHECK: module {
    print(builder.module_to_string(m))


fn test_empty_graph() raises:
    let builder = mobi.Builder()

    var m = builder.new_module()

    let ins = builder.new_arity()

    let outs = builder.new_arity()

    let unknown = builder.unknown_loc(m)
    _ = builder.new_graph(m, unknown, "empty_graph", ins, outs)

    # CHECK: builtin.module
    # CHECK: mo.graph
    # CHECK: sym_name = "empty_graph"
    print(builder.module_to_string(m))


fn test_identity_graph() raises:
    let builder = mobi.Builder()

    var m = builder.new_module()

    var ins = builder.new_arity()
    let scalarShape = DynamicVector[Int64]()
    let intScalar = builder.si(m, 32, scalarShape.data, len(scalarShape))
    builder.add_type(ins, intScalar)

    var outs = builder.new_arity()
    builder.add_type(outs, intScalar)

    let unknown = builder.unknown_loc(m)
    var g = builder.new_graph(m, unknown, "identity_graph", ins, outs)

    var rets = builder.new_tuple()
    let arg0 = builder.get_input(g, 0)
    builder.add_symbol(rets, arg0)

    let void = builder.new_arity()

    let attrs = builder.new_attr_map()

    _ = builder.new_op(g, unknown, "mo.output", rets, void, attrs)

    # CHECK: mo.graph @identity_graph
    # CHECK: mo.output
    print(builder.module_to_string(m))


fn test_tuple_copy() raises:
    let builder = mobi.Builder()

    var m = builder.new_module()

    var ins = builder.new_arity()
    let scalarShape = DynamicVector[Int64]()
    let intScalar = builder.si(m, 32, scalarShape.data, len(scalarShape))
    builder.add_type(ins, intScalar)

    var outs = builder.new_arity()
    builder.add_type(outs, intScalar)

    let unknown = builder.unknown_loc(m)
    var g = builder.new_graph(m, unknown, "tuple_copy_test", ins, outs)

    var t = builder.new_tuple()
    let arg0 = builder.get_input(g, 0)
    builder.add_symbol(t, arg0)

    var t2 = builder.new_tuple()
    let el = builder.get_symbol(t, 0)
    builder.add_symbol(t2, el)

    let void = builder.new_arity()

    let attrs = builder.new_attr_map()

    _ = builder.new_op(g, unknown, "mo.output", t2, void, attrs)

    # CHECK: mo.graph @tuple_copy_test
    # CHECK: mo.output
    print(builder.module_to_string(m))


fn test_const_op() raises:
    let builder = mobi.Builder()

    var m = builder.new_module()

    let void = builder.new_arity()

    var outs = builder.new_arity()
    let scalarShape = DynamicVector[Int64]()
    let intScalar = builder.si(m, 32, scalarShape.data, len(scalarShape))
    builder.add_type(outs, intScalar)

    let unknown = builder.unknown_loc(m)
    var g = builder.new_graph(m, unknown, "const_op", void, outs)

    var attrs = builder.new_attr_map()
    var val: Int32 = 1
    let a = builder.tensor_attr(
        m, "value", Pointer[Int32].address_of(val).bitcast[AnyType](), intScalar
    )
    builder.add_attr(attrs, a)

    let noIns = builder.new_tuple()
    let x = builder.new_op(g, unknown, "mo.constant", noIns, outs, attrs)

    var rets = builder.new_tuple()
    let x0 = builder.get_symbol(x, 0)
    builder.add_symbol(rets, x0)

    let noAttrs = builder.new_attr_map()

    _ = builder.new_op(g, unknown, "mo.output", rets, void, noAttrs)

    # CHECK: mo.graph @const_op
    # CHECK: mo.constant
    print(builder.module_to_string(m))


fn test_execution() raises:
    let builder = mobi.Builder()

    var m = builder.new_module()

    var ins = builder.new_arity()
    let scalarShape = DynamicVector[Int64]()
    let intScalar = builder.si(m, 32, scalarShape.data, len(scalarShape))
    builder.add_type(ins, intScalar)

    var outs = builder.new_arity()
    builder.add_type(outs, intScalar)

    let unknown = builder.unknown_loc(m)
    var g = builder.new_graph(m, unknown, "identity_graph", ins, outs)

    var rets = builder.new_tuple()
    let arg0 = builder.get_input(g, 0)
    builder.add_symbol(rets, arg0)

    let void = builder.new_arity()

    let attrs = builder.new_attr_map()

    _ = builder.new_op(g, unknown, "mo.output", rets, void, attrs)

    var rtArgs = builder.new_rt_tuple()
    builder.add_si32_rt_scalar(rtArgs, 1)
    var rtOuts = builder.new_rt_tuple()
    _ = builder.execute_module(m, rtArgs, rtOuts)

    # CHECK{LITERAL}: tensor([[1]], dtype=si32, shape=[])
    builder.debug_print_tensors(rtOuts)


fn main() raises:
    test_types()
    test_empty_module()
    test_empty_graph()
    test_identity_graph()
    test_tuple_copy()
    test_const_op()
    test_execution()
