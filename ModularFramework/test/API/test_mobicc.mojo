# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo -I %mobi_pkg_dir -D TEMP_DIR=%T %s | FileCheck %s
# RUN: mt -execute %T/module_to_bytecode.bin | FileCheck %s --check-prefix=CHECK-MTB

from sys.param_env import env_get_string
from testing import assert_equal

# import mobi
from mobi import mobicc


alias TEMP_DIR = Path(env_get_string["TEMP_DIR"]())


fn test_empty_module() raises:
    let m = mobicc.module_new()
    # CHECK: module {
    print(mobicc.module_to_string(m))


fn test_empty_graph() raises:
    var m = mobicc.module_new()

    let ins = mobicc.arity_new()

    let outs = mobicc.arity_new()

    let unknown = mobicc.loc_new_unknown(m)
    _ = mobicc.graph_new(m, unknown, "empty_graph", ins, outs)

    # CHECK: builtin.module
    # CHECK: mo.graph
    # CHECK: sym_name = "empty_graph"
    print(mobicc.module_to_string(m))


fn test_identity_graph() raises:
    var m = mobicc.module_new()

    var ins = mobicc.arity_new()
    let scalarShape = DynamicVector[Int64]()
    let intScalar = mobicc.tensor_type_new(
        m, mobicc.dtype_new(m, DType.int32), scalarShape, True
    )
    mobicc.arity_add_type(ins, intScalar)

    var outs = mobicc.arity_new()
    mobicc.arity_add_type(outs, intScalar)

    let unknown = mobicc.loc_new_unknown(m)
    let g = mobicc.graph_new(m, unknown, "identity_graph", ins, outs)

    var rets = mobicc.tuple_new()
    let arg0 = mobicc.graph_get_arg(g, 0)
    mobicc.tuple_add_symbol(rets, arg0)

    let void = mobicc.arity_new()

    let attrs = mobicc.attr_map_new()

    _ = mobicc.graph_new_op(g, unknown, "mo.output", rets, void, attrs)

    # CHECK: mo.graph @identity_graph
    # CHECK: mo.output
    print(mobicc.module_to_string(m))


fn test_tuple_copy() raises:
    var m = mobicc.module_new()

    var ins = mobicc.arity_new()
    let scalarShape = DynamicVector[Int64]()
    let intScalar = mobicc.tensor_type_new(
        m, mobicc.dtype_new(m, DType.int32), scalarShape, True
    )
    mobicc.arity_add_type(ins, intScalar)

    var outs = mobicc.arity_new()
    mobicc.arity_add_type(outs, intScalar)

    let unknown = mobicc.loc_new_unknown(m)
    let g = mobicc.graph_new(m, unknown, "tuple_copy_test", ins, outs)

    var t = mobicc.tuple_new()
    let arg0 = mobicc.graph_get_arg(g, 0)
    mobicc.tuple_add_symbol(t, arg0)

    var t2 = mobicc.tuple_new()
    let el = mobicc.tuple_get_symbol(t, 0)
    mobicc.tuple_add_symbol(t2, el)

    let void = mobicc.arity_new()

    let attrs = mobicc.attr_map_new()

    _ = mobicc.graph_new_op(g, unknown, "mo.output", t2, void, attrs)

    # CHECK: mo.graph @tuple_copy_test
    # CHECK: mo.output
    print(mobicc.module_to_string(m))


fn test_const_op() raises:
    var m = mobicc.module_new()

    let void = mobicc.arity_new()

    var outs = mobicc.arity_new()
    let scalarShape = DynamicVector[Int64]()
    let intScalar = mobicc.tensor_type_new(
        m, mobicc.dtype_new(m, DType.int32), scalarShape, True
    )
    mobicc.arity_add_type(outs, intScalar)

    let unknown = mobicc.loc_new_unknown(m)
    let g = mobicc.graph_new(m, unknown, "const_op", void, outs)

    var attrs = mobicc.attr_map_new()
    var val: Int32 = 1
    let a = mobicc.attr_new_tensor(
        m,
        "value",
        DTypePointer[DType.int32].address_of(val).bitcast[DType.invalid](),
        intScalar,
        is_owned=False,
    )
    mobicc.attr_map_add_attr(attrs, a)

    let noIns = mobicc.tuple_new()
    let x = mobicc.graph_new_op(g, unknown, "mo.constant", noIns, outs, attrs)

    var rets = mobicc.tuple_new()
    let x0 = mobicc.tuple_get_symbol(x, 0)
    mobicc.tuple_add_symbol(rets, x0)

    let noAttrs = mobicc.attr_map_new()

    _ = mobicc.graph_new_op(g, unknown, "mo.output", rets, void, noAttrs)

    # CHECK: mo.graph @const_op
    # CHECK: mo.constant
    print(mobicc.module_to_string(m))


fn test_bytecode_output() raises:
    var m = mobicc.module_new()

    let void = mobicc.arity_new()

    var outs = mobicc.arity_new()
    let scalarShape = DynamicVector[Int64]()
    let intScalar = mobicc.tensor_type_new(
        m, mobicc.dtype_new(m, DType.int32), scalarShape, True
    )
    mobicc.arity_add_type(outs, intScalar)

    let unknown = mobicc.loc_new_unknown(m)
    let g = mobicc.graph_new(m, unknown, "bytecode_output", void, outs)

    var attrs = mobicc.attr_map_new()
    var val: Int32 = 7
    let a = mobicc.attr_new_tensor(
        m,
        "value",
        DTypePointer[DType.int32].address_of(val).bitcast[DType.invalid](),
        intScalar,
        is_owned=False,
    )
    mobicc.attr_map_add_attr(attrs, a)

    let noIns = mobicc.tuple_new()
    let x = mobicc.graph_new_op(g, unknown, "mo.constant", noIns, outs, attrs)

    var rets = mobicc.tuple_new()
    let x0 = mobicc.tuple_get_symbol(x, 0)
    mobicc.tuple_add_symbol(rets, x0)

    let noAttrs = mobicc.attr_map_new()

    _ = mobicc.graph_new_op(g, unknown, "mo.output", rets, void, noAttrs)

    # CHECK: True
    # CHECK-MTB{LITERAL}: 'bytecode_output' returned tensor([[7]], dtype=si32, shape=[])
    print(
        mobicc.module_to_bytecode(
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
