# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .graph import Graph
from .mobicc import AttrMapPtr, ArityPtr, TuplePtr, SymbolPtr
from .symbol import Symbol
from .type import Type


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

    fn __init__(inout self, t: Type) raises:
        self.l = 1
        self.a = mobicc.new_arity()
        mobicc.add_type(self.a, t.t)

    fn __len__(self) -> Int:
        return self.l


struct Tup:
    """Tuple of symbols output by an operation."""

    # `Pointer` to the `M_Tuple` of symbols.
    var t: TuplePtr
    # Length of this `Tup`.
    var l: Int
    # Computation graph.
    var g: Graph

    fn __init__(inout self, t: TuplePtr, l: Int, g: Graph):
        self.t = t
        self.l = l
        self.g = g

    fn __init__(inout self, symbol: Symbol) raises:
        self.g = symbol.g
        self.l = 1
        self.t = mobicc.new_tuple()
        mobicc.add_symbol(self.t, symbol.s)

    # Request: @unique_value decorator for types
    # which acts like Value but only implements __moveinit__
    fn __moveinit__(inout self, owned existing: Self):
        self.t = existing.t ^
        self.l = existing.l
        self.g = existing.g

    fn __getitem__(self, pos: UInt32) raises -> Symbol:
        return Symbol(mobicc.get_symbol(self.t, pos), self.g)

    fn __len__(self) -> Int:
        return self.l
