# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import Pointer
from os import getenv
from sys import external_call
from tensor import Tensor
from sys.ffi import RTLD, DLHandle


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


struct Builder:
    """A builder for MO graphs. Wraps functins in the Builder C API."""

    var lib: DLHandle

    fn __init__(inout self) raises:
        let mof_lib = getenv("MODULAR_MOF_LIB_PATH")
        if mof_lib == "":
            raise "MODULAR_MOF_LIB_PATH must be set"

        # TODO: Any chance to just be able to use external_call?
        # Or does that only work with things statically linked into compiler?
        self.lib = DLHandle(mof_lib, RTLD.NOW | RTLD.GLOBAL)

    fn __copyinit__(inout self, other: Builder):
        self.lib = other.lib

    fn __del__(owned self):
        # TODO: self.lib._del_old() when we have refcounts.
        # For now, we leak this handle, in exchange for the ability to freely
        # copy Builders around.
        pass

    ### Attr helers

    fn tensor_attr(
        self,
        module: ModulePtr,
        name: StringRef,
        data: Pointer[AnyType],
        type: TypePtr,
    ) raises -> AttrPtr:
        return self.lib.get_function[
            fn (ModulePtr, StringRef, Pointer[AnyType], TypePtr) -> AttrPtr
        ]("MOBI_tensorAttr")(module, name, data, type)

    fn tensor_attr(
        self,
        module: ModulePtr,
        name: StringRef,
        data: DTypePointer[DType.invalid],
        type: TypePtr,
    ) raises -> AttrPtr:
        return self.lib.get_function[
            fn (
                ModulePtr, StringRef, DTypePointer[DType.invalid], TypePtr
            ) -> AttrPtr
        ]("MOBI_tensorAttr")(module, name, data, type)

    fn tensor_attr_from_file(
        self,
        module: ModulePtr,
        name: StringRef,
        file_name: StringRef,
        type: TypePtr,
    ) raises -> AttrPtr:
        return self.lib.get_function[
            fn (ModulePtr, StringRef, StringRef, TypePtr) -> AttrPtr
        ]("MOBI_tensorAttrFromFile")(module, name, file_name, type)

    fn new_attr_map(self) raises -> AttrMapPtr:
        return self.lib.get_function[fn () -> AttrMapPtr]("MOBI_newAttrMap")()

    fn add_attr(self, inout map: AttrMapPtr, a: AttrPtr) raises:
        return self.lib.get_function[fn (AttrMapPtr, AttrPtr) -> NoneType](
            "MOBI_addAttr"
        )(map, a)

    ### Graph helpers

    fn new_graph(
        self,
        inout module: ModulePtr,
        loc: LocPtr,
        name: StringRef,
        inTypes: ArityPtr,
        outTypes: ArityPtr,
    ) raises -> GraphPtr:
        return self.lib.get_function[
            fn (ModulePtr, LocPtr, StringRef, ArityPtr, ArityPtr) -> GraphPtr
        ]("MOBI_newGraph")(module, loc, name, inTypes, outTypes)

    fn get_input(
        self,
        graph: GraphPtr,
        pos: UInt32,
    ) raises -> SymbolPtr:
        return self.lib.get_function[fn (GraphPtr, UInt32) -> SymbolPtr](
            "MOBI_getInput"
        )(graph, pos)

    fn new_op(
        self,
        inout graph: GraphPtr,
        loc: LocPtr,
        name: StringRef,
        inputs: TuplePtr,
        outTypes: ArityPtr,
        attrs: AttrMapPtr,
    ) raises -> TuplePtr:
        return self.lib.get_function[
            fn (
                GraphPtr, LocPtr, StringRef, TuplePtr, ArityPtr, AttrMapPtr
            ) -> TuplePtr
        ]("MOBI_newOp")(graph, loc, name, inputs, outTypes, attrs)

    ### Location helpers

    fn unknown_loc(self, module: ModulePtr) raises -> LocPtr:
        return self.lib.get_function[fn (ModulePtr) -> LocPtr](
            "MOBI_unknownLoc"
        )(module)

    ### Module helpers

    fn new_module(self) raises -> ModulePtr:
        return self.lib.get_function[fn () -> ModulePtr]("MOBI_newModule")()

    fn debug_print_module(self, module: ModulePtr) raises:
        self.lib.get_function[fn (ModulePtr) -> NoneType](
            "MOBI_debugPrintModule"
        )(module)

    fn module_to_string(self, module: ModulePtr) raises -> String:
        let ret = self.lib.get_function[
            fn (ModulePtr) -> DTypePointer[DType.int8]
        ]("MOBI_moduleToString")(module)
        let sRet = String(ret)
        ret.free()
        return sRet

    ### Symbol helers

    fn new_tuple(self) raises -> TuplePtr:
        return self.lib.get_function[fn () -> TuplePtr]("MOBI_newTuple")()

    fn add_symbol(self, inout t: TuplePtr, s: SymbolPtr) raises:
        return self.lib.get_function[fn (TuplePtr, SymbolPtr) -> NoneType](
            "MOBI_addSymbol"
        )(t, s)

    fn get_symbol(self, t: TuplePtr, pos: UInt32) raises -> SymbolPtr:
        return self.lib.get_function[fn (TuplePtr, UInt32) -> SymbolPtr](
            "MOBI_getSymbol"
        )(t, pos)

    ### Type helers

    fn dynamic_dim(self, module: ModulePtr) raises -> Int64:
        return self.lib.get_function[fn (ModulePtr) -> Int64](
            "MOBI_dynamicDim"
        )(module)

    fn f32(
        self,
        module: ModulePtr,
        dims: Pointer[Int64],
        rank: Int64,
    ) raises -> TypePtr:
        return self.lib.get_function[
            fn (ModulePtr, Pointer[Int64], Int64) -> TypePtr
        ]("MOBI_f32")(module, dims, rank)

    fn ui(
        self,
        module: ModulePtr,
        width: UInt64,
        dims: Pointer[Int64],
        rank: Int64,
    ) raises -> TypePtr:
        return self.lib.get_function[
            fn (ModulePtr, UInt64, Pointer[Int64], Int64) -> TypePtr
        ]("MOBI_ui")(module, width, dims, rank)

    fn si(
        self,
        module: ModulePtr,
        width: UInt64,
        dims: Pointer[Int64],
        rank: Int64,
    ) raises -> TypePtr:
        return self.lib.get_function[
            fn (ModulePtr, UInt64, Pointer[Int64], Int64) -> TypePtr
        ]("MOBI_si")(module, width, dims, rank)

    fn bool(
        self,
        module: ModulePtr,
        dims: Pointer[Int64],
        rank: Int64,
    ) raises -> TypePtr:
        return self.lib.get_function[
            fn (ModulePtr, Pointer[Int64], Int64) -> TypePtr
        ]("MOBI_bool")(module, dims, rank)

    fn new_arity(self) raises -> ArityPtr:
        return self.lib.get_function[fn () -> ArityPtr]("MOBI_newArity")()

    fn add_type(self, inout arity: ArityPtr, t: TypePtr) raises:
        return self.lib.get_function[fn (ArityPtr, TypePtr) -> NoneType](
            "MOBI_addType"
        )(arity, t)

    fn type_is_float_tensor(self, t: TypePtr) raises -> Bool:
        return self.lib.get_function[fn (TypePtr) -> Bool](
            "MOBI_typeIsFloatTensor"
        )(t)

    fn type_is_si_tensor(self, t: TypePtr) raises -> Bool:
        return self.lib.get_function[fn (TypePtr) -> Bool](
            "MOBI_typeIsSITensor"
        )(t)

    fn type_is_ui_tensor(self, t: TypePtr) raises -> Bool:
        return self.lib.get_function[fn (TypePtr) -> Bool](
            "MOBI_typeIsUITensor"
        )(t)

    fn type_is_bool_tensor(self, t: TypePtr) raises -> Bool:
        return self.lib.get_function[fn (TypePtr) -> Bool](
            "MOBI_typeIsBoolTensor"
        )(t)

    fn type_is_tensor(self, t: TypePtr) raises -> Bool:
        return self.lib.get_function[fn (TypePtr) -> Bool]("MOBI_typeIsTensor")(
            t
        )

    fn tensor_dtype_width(self, t: TypePtr) raises -> UInt64:
        return self.lib.get_function[fn (TypePtr) -> UInt64](
            "MOBI_getTensorTypeDTypeWidth"
        )(t)

    fn tensor_rank(self, t: TypePtr) raises -> UInt64:
        return self.lib.get_function[fn (TypePtr) -> UInt64](
            "MOBI_getTensorTypeRank"
        )(t)

    fn tensor_dim(self, t: TypePtr, dim: UInt64) raises -> Int64:
        return self.lib.get_function[fn (TypePtr, UInt64) -> Int64](
            "MOBI_getTensorTypeDim"
        )(t, dim)

    ### Execution helpers

    # TODO: Merge with engine API

    fn new_rt_tuple(self) raises -> TensorTuplePtr:
        return self.lib.get_function[fn () -> TensorTuplePtr](
            "MOF_newTensorTuple"
        )()

    fn add_si32_rt_scalar(self, inout t: TensorTuplePtr, value: Int32) raises:
        self.lib.get_function[fn (TensorTuplePtr, Int32) -> NoneType](
            "MOF_addI32Scalar"
        )(t, value)

    fn add_f32_tensor(
        self,
        inout t: TensorTuplePtr,
        data: DTypePointer[DType.invalid],
        dims: Pointer[Int64],
        rank: Int64,
    ) raises:
        self.lib.get_function[
            fn (
                TensorTuplePtr,
                DTypePointer[DType.invalid],
                Pointer[Int64],
                Int64,
            ) -> NoneType
        ]("MOF_addF32Tensor")(t, data, dims, rank)

    fn add_f32_tensor_from_file(
        self,
        inout t: TensorTuplePtr,
        file_name: StringRef,
        dims: Pointer[Int64],
        rank: Int64,
    ) raises:
        self.lib.get_function[
            fn (TensorTuplePtr, StringRef, Pointer[Int64], Int64) -> NoneType
        ]("MOF_addF32TensorFromFile")(t, file_name, dims, rank)

    fn execute_module(
        self,
        module: ModulePtr,
        args: TensorTuplePtr,
        inout outs: TensorTuplePtr,
    ) raises -> UInt32:
        return self.lib.get_function[
            fn (ModulePtr, TensorTuplePtr, TensorTuplePtr) -> UInt32
        ]("MOF_executeModule")(module, args, outs)

    fn debug_print_tensors(self, t: TensorTuplePtr) raises:
        self.lib.get_function[fn (TensorTuplePtr) -> NoneType](
            "MOF_debugPrintTensors"
        )(t)
