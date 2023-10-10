# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import Pointer
from os import getenv
from sys import external_call
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


alias AttrPtr = Pointer[_Attr]
alias AttrMapPtr = Pointer[_AttrMap]
alias GraphPtr = Pointer[_Graph]
alias LocPtr = Pointer[_Loc]
alias ModulePtr = Pointer[_Module]
alias TuplePtr = Pointer[_Tuple]
alias SymbolPtr = Pointer[_Symbol]
alias ArityPtr = Pointer[_Arity]
alias TypePtr = Pointer[_Type]


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
        inout self,
        inout module: ModulePtr,
        name: StringRef,
        data: Pointer[AnyType],
        type: TypePtr,
    ) raises -> AttrPtr:
        return self.lib.get_function[
            fn (ModulePtr, StringRef, Pointer[AnyType], TypePtr) -> AttrPtr
        ]("MOBI_tensorAttr")(module, name, data, type)

    fn new_attr_map(inout self) raises -> AttrMapPtr:
        return self.lib.get_function[fn () -> AttrMapPtr]("MOBI_newAttrMap")()

    fn add_attr(inout self, inout map: AttrMapPtr, inout a: AttrPtr) raises:
        return self.lib.get_function[fn (AttrMapPtr, AttrPtr) -> NoneType](
            "MOBI_addAttr"
        )(map, a)

    ### Graph helpers

    fn new_graph(
        inout self,
        inout module: ModulePtr,
        inout loc: LocPtr,
        name: StringRef,
        inTypes: ArityPtr,
        outTypes: ArityPtr,
    ) raises -> GraphPtr:
        return self.lib.get_function[
            fn (ModulePtr, LocPtr, StringRef, ArityPtr, ArityPtr) -> GraphPtr
        ]("MOBI_newGraph")(module, loc, name, inTypes, outTypes)

    fn get_input(
        inout self,
        inout graph: GraphPtr,
        pos: UInt32,
    ) raises -> SymbolPtr:
        return self.lib.get_function[fn (GraphPtr, UInt32) -> SymbolPtr](
            "MOBI_getInput"
        )(graph, pos)

    fn new_op(
        inout self,
        inout graph: GraphPtr,
        inout loc: LocPtr,
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

    fn unknown_loc(inout self, inout module: ModulePtr) raises -> LocPtr:
        return self.lib.get_function[fn (ModulePtr) -> LocPtr](
            "MOBI_unknownLoc"
        )(module)

    ### Module helpers

    fn new_module(inout self) raises -> ModulePtr:
        return self.lib.get_function[fn () -> ModulePtr]("MOBI_newModule")()

    fn debug_print_module(inout self, inout module: ModulePtr) raises:
        self.lib.get_function[fn (ModulePtr) -> NoneType](
            "MOBI_debugPrintModule"
        )(module)

    ### Symbol helers

    fn new_tuple(inout self) raises -> TuplePtr:
        return self.lib.get_function[fn () -> TuplePtr]("MOBI_newTuple")()

    fn add_symbol(inout self, inout t: TuplePtr, inout s: SymbolPtr) raises:
        return self.lib.get_function[fn (TuplePtr, SymbolPtr) -> NoneType](
            "MOBI_addSymbol"
        )(t, s)

    fn get_symbol(
        inout self, inout t: TuplePtr, pos: UInt32
    ) raises -> SymbolPtr:
        return self.lib.get_function[fn (TuplePtr, UInt32) -> SymbolPtr](
            "MOBI_getSymbol"
        )(t, pos)

    ### Type helers

    fn si32_scalar(inout self, inout module: ModulePtr) raises -> TypePtr:
        return self.lib.get_function[fn (ModulePtr) -> TypePtr](
            "MOBI_si32Scalar"
        )(module)

    fn new_arity(inout self) raises -> ArityPtr:
        return self.lib.get_function[fn () -> ArityPtr]("MOBI_newArity")()

    fn add_type(inout self, inout arity: ArityPtr, inout t: TypePtr) raises:
        return self.lib.get_function[fn (ArityPtr, TypePtr) -> NoneType](
            "MOBI_addType"
        )(arity, t)
