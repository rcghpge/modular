# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.buffer import _MAX_RANK
from memory.unsafe import Pointer
from os import getenv
from sys import external_call
from tensor import Tensor, TensorShape
from sys.ffi import (
    RTLD,
    DLHandle,
    _get_dylib_function as _ffi_get_dylib_function,
)
from tensor import TensorSpec
from collections.vector import DynamicVector


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
fn _get_dylib_function[T: AnyRegType](name: StringRef) -> T:
    return _ffi_get_dylib_function["MOF_LIB", _init_dylib, _destroy_dylib, T](
        name
    )


@always_inline
fn MOBI_func[T: AnyRegType](name: StringRef) -> T:
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
    data: Pointer[AnyRegType],
    type: TypePtr,
    is_owned: Bool,
) -> AttrPtr:
    return MOBI_func[
        fn (ModulePtr, StringRef, Pointer[AnyRegType], TypePtr, Bool) -> AttrPtr
    ]("MOBI_tensorAttr")(module, name, data, type, is_owned)


fn tensor_attr(
    module: ModulePtr,
    name: StringRef,
    data: DTypePointer[DType.invalid],
    type: TypePtr,
    is_owned: Bool,
) -> AttrPtr:
    return MOBI_func[
        fn (
            ModulePtr, StringRef, DTypePointer[DType.invalid], TypePtr, Bool
        ) -> AttrPtr
    ]("MOBI_tensorAttr")(module, name, data, type, is_owned)


fn string_attr(
    module: ModulePtr,
    name: StringRef,
    value: StringRef,
) -> AttrPtr:
    return MOBI_func[fn (ModulePtr, StringRef, StringRef) -> AttrPtr](
        "MOBI_stringAttr"
    )(module, name, value)


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


fn get_graph_module(graph: GraphPtr) -> ModulePtr:
    return MOBI_func[fn (GraphPtr) -> ModulePtr]("MOBI_getGraphModule")(graph)


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


fn module_verify(module: ModulePtr) -> Bool:
    return MOBI_func[fn (ModulePtr) -> Bool]("MOBI_moduleVerify")(module)


fn module_to_string(module: ModulePtr) -> String:
    var len: Int64 = 0
    let ret = MOBI_func[fn (ModulePtr, Pointer[Int64]) -> Pointer[Int8]](
        "MOBI_moduleToString"
    )(module, Pointer[Int64].address_of(len))
    return String(ret, len.to_int())


fn module_to_bytecode(module: ModulePtr, file_name: String) -> Bool:
    return MOBI_func[fn (ModulePtr, StringRef) -> Bool](
        "MOBI_moduleToBytecode"
    )(module, file_name._strref_dangerous())


### Symbol helers


fn symbol_to_string(symbol: SymbolPtr) -> String:
    var len: Int64 = 0
    let ret = MOBI_func[fn (SymbolPtr, Pointer[Int64]) -> Pointer[Int8]](
        "MOBI_symbolToString"
    )(symbol, Pointer[Int64].address_of(len))
    return String(ret, len.to_int())


fn get_symbol_graph(symbol: SymbolPtr) -> GraphPtr:
    return MOBI_func[fn (SymbolPtr) -> GraphPtr]("MOBI_getSymbolGraph")(symbol)


fn new_tuple() -> TuplePtr:
    return MOBI_func[fn () -> TuplePtr]("MOBI_newTuple")()


fn tuple_size(tup: TuplePtr) -> Int:
    return MOBI_func[fn (TuplePtr) -> Int]("MOBI_tupleSize")(tup)


fn add_symbol(inout tup: TuplePtr, symbol: SymbolPtr):
    return MOBI_func[fn (TuplePtr, SymbolPtr) -> NoneType]("MOBI_addSymbol")(
        tup, symbol
    )


fn get_symbol(tup: TuplePtr, pos: UInt32) -> SymbolPtr:
    return MOBI_func[fn (TuplePtr, UInt32) -> SymbolPtr]("MOBI_getSymbol")(
        tup, pos
    )


### Type helers


fn type_to_string(type: TypePtr) -> String:
    var len: Int64 = 0
    let ret = MOBI_func[fn (TypePtr, Pointer[Int64]) -> Pointer[Int8]](
        "MOBI_typeToString"
    )(type, Pointer[Int64].address_of(len))
    return String(ret, len.to_int())


fn new_dtype(m: ModulePtr, dtype: DType) -> TypePtr:
    return MOBI_func[fn (ModulePtr, UInt8) -> TypePtr]("MOBI_newDType")(
        m, dtype._as_i8()
    )


fn new_tensor_type(
    m: ModulePtr, dtype: TypePtr, dims: DynamicVector[Int64], ranked: Bool
) -> TypePtr:
    return MOBI_func[
        fn (ModulePtr, TypePtr, Bool, Pointer[Int64], Int32) -> TypePtr
    ]("MOBI_newTensorType")(
        m, dtype, ranked, Pointer[Int64](dims.data.value), len(dims)
    )


fn get_tensor_dtype(s: SymbolPtr) -> DType:
    let dtype = MOBI_func[fn (SymbolPtr) -> UInt8]("MOBI_getTensorDType")(s)
    return DType._from_ui8(dtype.value)


fn get_tensor_shape(s: SymbolPtr) -> DynamicVector[Int64]:
    var rank: Int32 = 0
    let dims = MOBI_func[fn (SymbolPtr, Pointer[Int32]) -> Pointer[Int64]](
        "MOBI_getTensorShape"
    )(s, Pointer.address_of(rank))
    var dimsVec = DynamicVector[Int64]()
    for i in range(rank):
        dimsVec.append(dims[i])
    return dimsVec


fn is_tensor_ranked(s: SymbolPtr) -> Bool:
    return MOBI_func[fn (SymbolPtr) -> Bool]("MOBI_isTensorRanked")(s)


fn type_from_spec(m: ModulePtr, spec: TensorSpec) -> TypePtr:
    return MOBI_func[fn (ModulePtr, TensorSpec) -> TypePtr](
        "MOBI_typeFromSpec"
    )(m, spec)


fn spec_from_type(t: TypePtr) -> TensorSpec:
    let dims = DTypePointer[DType.int64].alloc(_MAX_RANK)
    let rank = DTypePointer[DType.int64].alloc(1)

    let dtype = MOBI_func[
        fn (
            TypePtr, DTypePointer[DType.int64], DTypePointer[DType.int64]
        ) -> UInt8
    ]("MOBI_specFromType")(t, dims, rank)

    var dims_vec = DynamicVector[Int]()
    for i in range(rank.load().to_int()):
        dims_vec.push_back(dims[i].to_int())

    let result_spec = TensorSpec(DType._from_ui8(dtype.value), dims_vec)

    rank.free()
    dims.free()

    return result_spec


fn dynamic_dim() -> Int64:
    return MOBI_func[fn () -> Int64]("MOBI_dynamicDim")()


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


# TODO: These specialized helpers are a stop-gap until we have sufficient
# type manipulation primitives to write the logic in Mojo.
# Note: Most type manipulation logic will become obsolete in HiMO, where
# all output types are inferred.
fn arg_reduction_type(
    module: ModulePtr,
    input: TypePtr,
    axis: Int64,
) -> TypePtr:
    return MOBI_func[fn (ModulePtr, TypePtr, Int64) -> TypePtr](
        "MOBI_argReductionType"
    )(module, input, axis)


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


fn get_symbol_type(s: SymbolPtr) -> TypePtr:
    return MOBI_func[fn (SymbolPtr) -> TypePtr]("MOBI_getSymbolType")(s)


fn del_type(t: TypePtr) -> None:
    return MOBI_func[fn (TypePtr) -> None]("MOBI_freeType")(t)
