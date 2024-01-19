# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo -D TEMP_DIR=%T %s | FileCheck %s
# RUN: mt -execute %T/module_to_bytecode.bin | FileCheck %s --check-prefix=CHECK-MTB

from sys.param_env import env_get_string

from max.graph import capi


alias TEMP_DIR = Path(env_get_string["TEMP_DIR"]())


fn test_empty_module() raises:
    let m = capi.module_new()
    # CHECK: module {
    print(capi.module_to_string(m))


fn test_empty_graph() raises:
    var m = capi.module_new()

    let ins = capi.arity_new()

    let outs = capi.arity_new()

    let unknown = capi.loc_new_unknown(m)
    _ = capi.graph_new(m, unknown, "empty_graph", ins, outs)

    # CHECK: builtin.module
    # CHECK: mo.graph
    # CHECK: sym_name = "empty_graph"
    print(capi.module_to_string(m))


fn test_identity_graph() raises:
    var m = capi.module_new()

    var ins = capi.arity_new()
    let scalarShape = DynamicVector[Int64]()
    let intScalar = capi.tensor_type_new(
        m, capi.dtype_new(m, DType.int32), scalarShape, True
    )
    capi.arity_add_type(ins, intScalar)

    var outs = capi.arity_new()
    capi.arity_add_type(outs, intScalar)

    let unknown = capi.loc_new_unknown(m)
    let g = capi.graph_new(m, unknown, "identity_graph", ins, outs)

    var rets = capi.tuple_new()
    let arg0 = capi.graph_get_arg(g, 0)
    capi.tuple_add_symbol(rets, arg0)

    let void = capi.arity_new()

    let attrs = capi.attr_map_new()

    _ = capi.graph_new_op(g, unknown, "mo.output", rets, void, attrs)

    # CHECK: mo.graph @identity_graph
    # CHECK: mo.output
    print(capi.module_to_string(m))


fn test_tuple_copy() raises:
    var m = capi.module_new()

    var ins = capi.arity_new()
    let scalarShape = DynamicVector[Int64]()
    let intScalar = capi.tensor_type_new(
        m, capi.dtype_new(m, DType.int32), scalarShape, True
    )
    capi.arity_add_type(ins, intScalar)

    var outs = capi.arity_new()
    capi.arity_add_type(outs, intScalar)

    let unknown = capi.loc_new_unknown(m)
    let g = capi.graph_new(m, unknown, "tuple_copy_test", ins, outs)

    var t = capi.tuple_new()
    let arg0 = capi.graph_get_arg(g, 0)
    capi.tuple_add_symbol(t, arg0)

    var t2 = capi.tuple_new()
    let el = capi.tuple_get_symbol(t, 0)
    capi.tuple_add_symbol(t2, el)

    let void = capi.arity_new()

    let attrs = capi.attr_map_new()

    _ = capi.graph_new_op(g, unknown, "mo.output", t2, void, attrs)

    # CHECK: mo.graph @tuple_copy_test
    # CHECK: mo.output
    print(capi.module_to_string(m))


fn test_const_op() raises:
    var m = capi.module_new()

    let void = capi.arity_new()

    var outs = capi.arity_new()
    let scalarShape = DynamicVector[Int64]()
    let intScalar = capi.tensor_type_new(
        m, capi.dtype_new(m, DType.int32), scalarShape, True
    )
    capi.arity_add_type(outs, intScalar)

    let unknown = capi.loc_new_unknown(m)
    let g = capi.graph_new(m, unknown, "const_op", void, outs)

    var attrs = capi.attr_map_new()
    var val: Int32 = 1
    let a = capi.attr_new_tensor(
        m,
        "value",
        DTypePointer[DType.int32].address_of(val).bitcast[DType.invalid](),
        intScalar,
        is_owned=False,
    )
    capi.attr_map_add_attr(attrs, a)

    let noIns = capi.tuple_new()
    let x = capi.graph_new_op(g, unknown, "mo.constant", noIns, outs, attrs)

    var rets = capi.tuple_new()
    let x0 = capi.tuple_get_symbol(x, 0)
    capi.tuple_add_symbol(rets, x0)

    let noAttrs = capi.attr_map_new()

    _ = capi.graph_new_op(g, unknown, "mo.output", rets, void, noAttrs)

    # CHECK: mo.graph @const_op
    # CHECK: mo.constant
    print(capi.module_to_string(m))


fn test_bytecode_output() raises:
    var m = capi.module_new()

    let void = capi.arity_new()

    var outs = capi.arity_new()
    let scalarShape = DynamicVector[Int64]()
    let intScalar = capi.tensor_type_new(
        m, capi.dtype_new(m, DType.int32), scalarShape, True
    )
    capi.arity_add_type(outs, intScalar)

    let unknown = capi.loc_new_unknown(m)
    let g = capi.graph_new(m, unknown, "bytecode_output", void, outs)

    var attrs = capi.attr_map_new()
    var val: Int32 = 7
    let a = capi.attr_new_tensor(
        m,
        "value",
        DTypePointer[DType.int32].address_of(val).bitcast[DType.invalid](),
        intScalar,
        is_owned=False,
    )
    capi.attr_map_add_attr(attrs, a)

    let noIns = capi.tuple_new()
    let x = capi.graph_new_op(g, unknown, "mo.constant", noIns, outs, attrs)

    var rets = capi.tuple_new()
    let x0 = capi.tuple_get_symbol(x, 0)
    capi.tuple_add_symbol(rets, x0)

    let noAttrs = capi.attr_map_new()

    _ = capi.graph_new_op(g, unknown, "mo.output", rets, void, noAttrs)

    # CHECK: True
    # CHECK-MTB{LITERAL}: 'bytecode_output' returned tensor([[7]], dtype=si32, shape=[])
    print(
        capi.module_to_bytecode(
            m, (TEMP_DIR / "module_to_bytecode.bin").__str__()
        )
    )


fn main() raises:
    test_empty_module()
    test_empty_graph()
    test_identity_graph()
    test_tuple_copy()
    test_const_op()
    test_bytecode_output()
