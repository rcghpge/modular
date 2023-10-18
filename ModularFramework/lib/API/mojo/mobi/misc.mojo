# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .mobicc import AttrMapPtr, ArityPtr, TuplePtr, SymbolPtr


struct AttrMap:
    var m: AttrMapPtr
    var l: Int

    fn __init__(inout self, m: AttrMapPtr, l: Int):
        self.m = m
        self.l = l

    fn __len__(self) -> Int:
        return self.l


struct Arity:
    var a: ArityPtr
    var l: Int

    fn __init__(inout self, a: ArityPtr, l: Int):
        self.a = a
        self.l = l

    fn __len__(self) -> Int:
        return self.l


struct Tup:
    var t: TuplePtr
    var l: Int
    var builder: Builder

    fn __init__(inout self, t: TuplePtr, l: Int, builder: Builder):
        self.t = t
        self.l = l
        self.builder = builder

    fn __getitem__(inout self, pos: UInt32) raises -> SymbolPtr:
        return self.builder.get_symbol(self.t, pos)

    fn __len__(self) -> Int:
        return self.l
