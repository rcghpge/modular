# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo -I %mobi_pkg_dir %s | FileCheck %s

import mobi
from mobi import mobicc


fn test_types() raises:
    let m = mobicc.new_module()
    let scalarShape = DynamicVector[Int64]()
    let intScalar = mobicc.si(m, 32, scalarShape.data, len(scalarShape))

    # CHECK: True
    print(mobicc.type_is_tensor(intScalar))
    # CHECK: True
    print(mobicc.type_is_si_tensor(intScalar))
    # CHECK: False
    print(mobicc.type_is_ui_tensor(intScalar))
    # CHECK: False
    print(mobicc.type_is_bool_tensor(intScalar))
    # CHECK: 32
    print(mobicc.tensor_dtype_width(intScalar))
    # CHECK: 0
    print(mobicc.tensor_rank(intScalar))

    var boolShape = DynamicVector[Int64]()
    boolShape.push_back(2)
    boolShape.push_back(3)
    let boolTensor = mobicc.bool(m, boolShape.data, len(boolShape))

    # CHECK: True
    print(mobicc.type_is_tensor(boolTensor))
    # CHECK: False
    print(mobicc.type_is_si_tensor(boolTensor))
    # CHECK: False
    print(mobicc.type_is_ui_tensor(boolTensor))
    # CHECK: True
    print(mobicc.type_is_bool_tensor(boolTensor))
    # CHECK: 8
    print(mobicc.tensor_dtype_width(boolTensor))
    # CHECK: 2
    print(mobicc.tensor_rank(boolTensor))
    # CHECK: 2
    print(mobicc.tensor_dim(boolTensor, 0))
    # CHECK: 3
    print(mobicc.tensor_dim(boolTensor, 1))

    scalarShape._del_old()
    boolShape._del_old()


fn test_empty_module() raises:
    let m = mobicc.new_module()
    # CHECK: module {
    print(mobicc.module_to_string(m))


fn test_empty_graph() raises:
    var m = mobicc.new_module()

    let ins = mobicc.new_arity()

    let outs = mobicc.new_arity()

    let unknown = mobicc.unknown_loc(m)
    _ = mobicc.new_graph(m, unknown, "empty_graph", ins, outs)

    # CHECK: builtin.module
    # CHECK: mo.graph
    # CHECK: sym_name = "empty_graph"
    print(mobicc.module_to_string(m))


fn test_identity_graph() raises:
    var m = mobicc.new_module()

    var ins = mobicc.new_arity()
    let scalarShape = DynamicVector[Int64]()
    let intScalar = mobicc.si(m, 32, scalarShape.data, len(scalarShape))
    mobicc.add_type(ins, intScalar)

    var outs = mobicc.new_arity()
    mobicc.add_type(outs, intScalar)

    let unknown = mobicc.unknown_loc(m)
    let g = mobicc.new_graph(m, unknown, "identity_graph", ins, outs)

    var rets = mobicc.new_tuple()
    let arg0 = mobicc.get_input(g, 0)
    mobicc.add_symbol(rets, arg0)

    let void = mobicc.new_arity()

    let attrs = mobicc.new_attr_map()

    _ = mobicc.new_op(g, unknown, "mo.output", rets, void, attrs)

    # CHECK: mo.graph @identity_graph
    # CHECK: mo.output
    print(mobicc.module_to_string(m))


fn test_tuple_copy() raises:
    var m = mobicc.new_module()

    var ins = mobicc.new_arity()
    let scalarShape = DynamicVector[Int64]()
    let intScalar = mobicc.si(m, 32, scalarShape.data, len(scalarShape))
    mobicc.add_type(ins, intScalar)

    var outs = mobicc.new_arity()
    mobicc.add_type(outs, intScalar)

    let unknown = mobicc.unknown_loc(m)
    let g = mobicc.new_graph(m, unknown, "tuple_copy_test", ins, outs)

    var t = mobicc.new_tuple()
    let arg0 = mobicc.get_input(g, 0)
    mobicc.add_symbol(t, arg0)

    var t2 = mobicc.new_tuple()
    let el = mobicc.get_symbol(t, 0)
    mobicc.add_symbol(t2, el)

    let void = mobicc.new_arity()

    let attrs = mobicc.new_attr_map()

    _ = mobicc.new_op(g, unknown, "mo.output", t2, void, attrs)

    # CHECK: mo.graph @tuple_copy_test
    # CHECK: mo.output
    print(mobicc.module_to_string(m))


fn test_const_op() raises:
    var m = mobicc.new_module()

    let void = mobicc.new_arity()

    var outs = mobicc.new_arity()
    let scalarShape = DynamicVector[Int64]()
    let intScalar = mobicc.si(m, 32, scalarShape.data, len(scalarShape))
    mobicc.add_type(outs, intScalar)

    let unknown = mobicc.unknown_loc(m)
    let g = mobicc.new_graph(m, unknown, "const_op", void, outs)

    var attrs = mobicc.new_attr_map()
    var val: Int32 = 1
    let a = mobicc.tensor_attr(
        m, "value", Pointer[Int32].address_of(val).bitcast[AnyType](), intScalar
    )
    mobicc.add_attr(attrs, a)

    let noIns = mobicc.new_tuple()
    let x = mobicc.new_op(g, unknown, "mo.constant", noIns, outs, attrs)

    var rets = mobicc.new_tuple()
    let x0 = mobicc.get_symbol(x, 0)
    mobicc.add_symbol(rets, x0)

    let noAttrs = mobicc.new_attr_map()

    _ = mobicc.new_op(g, unknown, "mo.output", rets, void, noAttrs)

    # CHECK: mo.graph @const_op
    # CHECK: mo.constant
    print(mobicc.module_to_string(m))


fn test_execution() raises:
    var m = mobicc.new_module()

    var ins = mobicc.new_arity()
    let scalarShape = DynamicVector[Int64]()
    let intScalar = mobicc.si(m, 32, scalarShape.data, len(scalarShape))
    mobicc.add_type(ins, intScalar)

    var outs = mobicc.new_arity()
    mobicc.add_type(outs, intScalar)

    let unknown = mobicc.unknown_loc(m)
    let g = mobicc.new_graph(m, unknown, "identity_graph", ins, outs)

    var rets = mobicc.new_tuple()
    let arg0 = mobicc.get_input(g, 0)
    mobicc.add_symbol(rets, arg0)

    let void = mobicc.new_arity()

    let attrs = mobicc.new_attr_map()

    _ = mobicc.new_op(g, unknown, "mo.output", rets, void, attrs)

    var rtArgs = mobicc.new_rt_tuple()
    mobicc.add_si32_rt_scalar(rtArgs, 1)
    var rtOuts = mobicc.new_rt_tuple()
    _ = mobicc.execute_module(m, rtArgs, rtOuts)

    # CHECK{LITERAL}: tensor([[1]], dtype=si32, shape=[])
    mobicc.debug_print_tensors(rtOuts)


fn main() raises:
    test_types()
    test_empty_module()
    test_empty_graph()
    test_identity_graph()
    test_tuple_copy()
    test_const_op()
    test_execution()
