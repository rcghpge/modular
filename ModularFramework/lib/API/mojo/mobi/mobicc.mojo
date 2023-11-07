# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import Pointer
from os import getenv
from sys import external_call
from tensor import Tensor, TensorShape
from sys.ffi import (
    RTLD,
    DLHandle,
    _get_dylib_function as _ffi_get_dylib_function,
)


struct _Attr:
    """Opaque data type mapping to M_Attr in the Builder C API."""

    pass


struct _AttrMap:
    """Opaque data type mapping to M_AttrMap in the Builder C API."""

    pass


struct _Graph:
    """Opaque data type mapping to M_Graph in the Builder C API."""

    pass


struct _Loc:
    """Opaque data type mapping to M_Loc in the Builder C API."""

    pass


struct _Module:
    """Opaque data type mapping to M_Module in the Builder C API."""

    pass


struct _Tuple:
    """Opaque data type mapping to M_Tuple in the Builder C API."""

    pass


struct _Symbol:
    """Opaque data type mapping to M_Symbol in the Builder C API."""

    pass


struct _Arity:
    """Opaque data type mapping to M_Arity in the Builder C API."""

    pass


struct _Type:
    """Opaque data type mapping to M_Type in the Builder C API."""

    pass


struct _TensorTuple:
    """Opaque data type mapping to M_TensorTuple in the MOF C API."""

    pass


alias AttrPtr = Pointer[_Attr]
alias AttrMapPtr = Pointer[_AttrMap]
alias GraphPtr = Pointer[_Graph]
alias LocPtr = Pointer[_Loc]
alias ModulePtr = Pointer[_Module]
alias TuplePtr = Pointer[_Tuple]
alias SymbolPtr = Pointer[_Symbol]
alias ArityPtr = Pointer[_Arity]
alias TypePtr = Pointer[_Type]
alias TensorTuplePtr = Pointer[_TensorTuple]


# Init fns inspired by gpu.host._utils
fn _init_dylib(ignored: Pointer[NoneType]) -> Pointer[NoneType]:
    let _mof_lib_path = getenv("MODULAR_MOF_LIB_PATH")
    let ptr = Pointer[DLHandle].alloc(1)
    __get_address_as_lvalue(ptr.address) = DLHandle(
        _mof_lib_path, RTLD.NOW | RTLD.GLOBAL
    )
    return ptr.bitcast[NoneType]()


fn _destroy_dylib(ptr: Pointer[NoneType]):
    __get_address_as_lvalue(ptr.bitcast[DLHandle]().address)._del_old()
    ptr.free()


@always_inline
fn _get_dylib_function[T: AnyType](name: StringRef) -> T:
    return _ffi_get_dylib_function["MOF_LIB", _init_dylib, _destroy_dylib, T](
        name
    )


@always_inline
fn MOBI_func[T: AnyType](name: StringRef) -> T:
    """`dlsym()`s a function called `name` in shared library `.lib`."""
    var f = _get_dylib_function[T](name)
    let ptr = Pointer.address_of(f).bitcast[Pointer[NoneType]]().load()
    if not ptr:
        print("cannot load " + String(name) + " from shared library")
        trap()
    return f


### Attr helpers


fn tensor_attr(
    module: ModulePtr,
    name: StringRef,
    data: Pointer[AnyType],
    type: TypePtr,
) -> AttrPtr:
    return MOBI_func[
        fn (ModulePtr, StringRef, Pointer[AnyType], TypePtr) -> AttrPtr
    ]("MOBI_tensorAttr")(module, name, data, type)


fn tensor_attr(
    module: ModulePtr,
    name: StringRef,
    data: DTypePointer[DType.invalid],
    type: TypePtr,
) -> AttrPtr:
    return MOBI_func[
        fn (
            ModulePtr, StringRef, DTypePointer[DType.invalid], TypePtr
        ) -> AttrPtr
    ]("MOBI_tensorAttr")(module, name, data, type)


fn tensor_attr_from_file(
    module: ModulePtr,
    name: StringRef,
    file_name: StringRef,
    type: TypePtr,
) -> AttrPtr:
    return MOBI_func[fn (ModulePtr, StringRef, StringRef, TypePtr) -> AttrPtr](
        "MOBI_tensorAttrFromFile"
    )(module, name, file_name, type)


fn new_attr_map() -> AttrMapPtr:
    return MOBI_func[fn () -> AttrMapPtr]("MOBI_newAttrMap")()


fn add_attr(inout map: AttrMapPtr, a: AttrPtr):
    return MOBI_func[fn (AttrMapPtr, AttrPtr) -> NoneType]("MOBI_addAttr")(
        map, a
    )


### Graph helpers


fn new_graph(
    inout module: ModulePtr,
    loc: LocPtr,
    name: StringRef,
    inTypes: ArityPtr,
    outTypes: ArityPtr,
) -> GraphPtr:
    return MOBI_func[
        fn (ModulePtr, LocPtr, StringRef, ArityPtr, ArityPtr) -> GraphPtr
    ]("MOBI_newGraph")(module, loc, name, inTypes, outTypes)


fn get_input(
    graph: GraphPtr,
    pos: UInt32,
) -> SymbolPtr:
    return MOBI_func[fn (GraphPtr, UInt32) -> SymbolPtr]("MOBI_getInput")(
        graph, pos
    )


fn new_op(
    graph: GraphPtr,
    loc: LocPtr,
    name: StringRef,
    inputs: TuplePtr,
    outTypes: ArityPtr,
    attrs: AttrMapPtr,
) -> TuplePtr:
    return MOBI_func[
        fn (
            GraphPtr, LocPtr, StringRef, TuplePtr, ArityPtr, AttrMapPtr
        ) -> TuplePtr
    ]("MOBI_newOp")(graph, loc, name, inputs, outTypes, attrs)


### Location helpers


fn unknown_loc(module: ModulePtr) -> LocPtr:
    return MOBI_func[fn (ModulePtr) -> LocPtr]("MOBI_unknownLoc")(module)


### Module helpers


fn new_module() -> ModulePtr:
    return MOBI_func[fn () -> ModulePtr]("MOBI_newModule")()


fn debug_print_module(module: ModulePtr):
    MOBI_func[fn (ModulePtr) -> NoneType]("MOBI_debugPrintModule")(module)


fn module_to_string(module: ModulePtr) -> String:
    let ret = MOBI_func[fn (ModulePtr) -> DTypePointer[DType.int8]](
        "MOBI_moduleToString"
    )(module)
    let sRet = String(ret)
    ret.free()
    return sRet


### Symbol helers


fn new_tuple() -> TuplePtr:
    return MOBI_func[fn () -> TuplePtr]("MOBI_newTuple")()


fn add_symbol(inout t: TuplePtr, s: SymbolPtr):
    return MOBI_func[fn (TuplePtr, SymbolPtr) -> NoneType]("MOBI_addSymbol")(
        t, s
    )


fn get_symbol(t: TuplePtr, pos: UInt32) -> SymbolPtr:
    return MOBI_func[fn (TuplePtr, UInt32) -> SymbolPtr]("MOBI_getSymbol")(
        t, pos
    )


### Type helers


fn dynamic_dim(module: ModulePtr) -> Int64:
    return MOBI_func[fn (ModulePtr) -> Int64]("MOBI_dynamicDim")(module)


fn f32(
    module: ModulePtr,
    dims: Pointer[Int64],
    rank: Int64,
) -> TypePtr:
    return MOBI_func[fn (ModulePtr, Pointer[Int64], Int64) -> TypePtr](
        "MOBI_f32"
    )(module, dims, rank)


fn ui(
    module: ModulePtr,
    width: UInt64,
    dims: Pointer[Int64],
    rank: Int64,
) -> TypePtr:
    return MOBI_func[fn (ModulePtr, UInt64, Pointer[Int64], Int64) -> TypePtr](
        "MOBI_ui"
    )(module, width, dims, rank)


fn si(
    module: ModulePtr,
    width: UInt64,
    dims: Pointer[Int64],
    rank: Int64,
) -> TypePtr:
    return MOBI_func[fn (ModulePtr, UInt64, Pointer[Int64], Int64) -> TypePtr](
        "MOBI_si"
    )(module, width, dims, rank)


fn bool(
    module: ModulePtr,
    dims: Pointer[Int64],
    rank: Int64,
) -> TypePtr:
    return MOBI_func[fn (ModulePtr, Pointer[Int64], Int64) -> TypePtr](
        "MOBI_bool"
    )(module, dims, rank)


fn new_arity() -> ArityPtr:
    return MOBI_func[fn () -> ArityPtr]("MOBI_newArity")()


fn add_type(inout arity: ArityPtr, t: TypePtr):
    return MOBI_func[fn (ArityPtr, TypePtr) -> NoneType]("MOBI_addType")(
        arity, t
    )


fn type_is_float_tensor(t: TypePtr) -> Bool:
    return MOBI_func[fn (TypePtr) -> Bool]("MOBI_typeIsFloatTensor")(t)


fn type_is_si_tensor(t: TypePtr) -> Bool:
    return MOBI_func[fn (TypePtr) -> Bool]("MOBI_typeIsSITensor")(t)


fn type_is_ui_tensor(t: TypePtr) -> Bool:
    return MOBI_func[fn (TypePtr) -> Bool]("MOBI_typeIsUITensor")(t)


fn type_is_bool_tensor(t: TypePtr) -> Bool:
    return MOBI_func[fn (TypePtr) -> Bool]("MOBI_typeIsBoolTensor")(t)


fn type_is_tensor(t: TypePtr) -> Bool:
    return MOBI_func[fn (TypePtr) -> Bool]("MOBI_typeIsTensor")(t)


fn tensor_dtype_width(t: TypePtr) -> UInt64:
    return MOBI_func[fn (TypePtr) -> UInt64]("MOBI_getTensorTypeDTypeWidth")(t)


fn tensor_rank(t: TypePtr) -> UInt64:
    return MOBI_func[fn (TypePtr) -> UInt64]("MOBI_getTensorTypeRank")(t)


fn tensor_dim(t: TypePtr, dim: UInt64) -> Int64:
    return MOBI_func[fn (TypePtr, UInt64) -> Int64]("MOBI_getTensorTypeDim")(
        t, dim
    )


fn typeof(s: SymbolPtr) -> TypePtr:
    """The result must be freed by the caller using `del_type()`."""
    return MOBI_func[fn (SymbolPtr) -> TypePtr]("MOBI_typeOfSymbol")(s)


fn del_type(t: TypePtr) -> None:
    return MOBI_func[fn (TypePtr) -> None]("MOBI_freeType")(t)


### Execution helpers

# TODO: Merge with engine API


fn new_rt_tuple() -> TensorTuplePtr:
    return MOBI_func[fn () -> TensorTuplePtr]("MOF_newTensorTuple")()


fn add_si32_rt_scalar(inout t: TensorTuplePtr, value: Int32):
    MOBI_func[fn (TensorTuplePtr, Int32) -> NoneType]("MOF_addI32Scalar")(
        t, value
    )


fn add_f32_tensor(inout out: TensorTuplePtr, inout tensor: Tensor):
    # Mojo` TensorShape` is ABI compatible with `M::TensorShape`.
    # See Kernels/unittests/TensorTest.{cpp,mojo}.
    # So just pass a pointer to `shape` across the shared library boundary.
    let shape_ptr = Pointer.address_of(tensor._spec.shape._rep).bitcast[
        TensorShape
    ]()
    let data_ptr = tensor.data().bitcast[DType.invalid]()
    MOBI_func[
        fn (
            TensorTuplePtr,
            DTypePointer[DType.invalid],
            Pointer[TensorShape],
        ) -> NoneType
    ]("MOF_addF32Tensor")(out, data_ptr, shape_ptr)


fn add_f32_tensor_from_file(
    inout t: TensorTuplePtr,
    file_name: StringRef,
    dims: Pointer[Int64],
    rank: Int64,
):
    MOBI_func[
        fn (TensorTuplePtr, StringRef, Pointer[Int64], Int64) -> NoneType
    ]("MOF_addF32TensorFromFile")(t, file_name, dims, rank)


fn execute_module(
    module: ModulePtr,
    args: TensorTuplePtr,
    inout outs: TensorTuplePtr,
) -> UInt32:
    return MOBI_func[fn (ModulePtr, TensorTuplePtr, TensorTuplePtr) -> UInt32](
        "MOF_executeModule"
    )(module, args, outs)


fn debug_print_tensors(t: TensorTuplePtr):
    MOBI_func[fn (TensorTuplePtr) -> NoneType]("MOF_debugPrintTensors")(t)
