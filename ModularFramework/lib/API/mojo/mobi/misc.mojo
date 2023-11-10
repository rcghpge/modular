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

    fn __init__(inout self, t: Type):
        self.l = 1
        self.a = mobicc.new_arity()
        mobicc.add_type(self.a, t.type_ptr())

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

    fn __init__(inout self, symbols: VariadicList[Symbol]) raises:
        if len(symbols) < 1:
            raise "Must pass at least 1 symbol"
        self.g = symbols[0].g
        self.l = len(symbols)
        self.t = mobicc.new_tuple()
        for symbol in symbols:
            mobicc.add_symbol(self.t, symbol.s)

    fn __init__(inout self, symbols: DynamicVector[Symbol]) raises:
        if len(symbols) < 1:
            raise "Must pass at least 1 symbol"
        self.g = symbols[0].g
        self.l = len(symbols)
        self.t = mobicc.new_tuple()
        for i in range(len(symbols)):
            mobicc.add_symbol(self.t, symbols[i].s)

    fn __init__(inout self, symbol: Symbol):
        self.g = symbol.g
        self.l = 1
        self.t = mobicc.new_tuple()
        mobicc.add_symbol(self.t, symbol.s)

    fn __init__(inout self, symbols: (Symbol, Symbol)):
        let s0 = symbols.get[0, Symbol]()
        let s1 = symbols.get[1, Symbol]()
        self.g = s0.g
        self.l = 2
        self.t = mobicc.new_tuple()
        mobicc.add_symbol(self.t, s0.s)
        mobicc.add_symbol(self.t, s1.s)

    fn __init__(inout self, symbols: (Symbol, Symbol, Symbol)):
        let s0 = symbols.get[0, Symbol]()
        let s1 = symbols.get[1, Symbol]()
        let s2 = symbols.get[2, Symbol]()
        self.g = s0.g
        self.l = 3
        self.t = mobicc.new_tuple()
        mobicc.add_symbol(self.t, s0.s)
        mobicc.add_symbol(self.t, s1.s)
        mobicc.add_symbol(self.t, s2.s)

    fn __init__(inout self, symbols: (Symbol, Symbol, Symbol, Symbol)):
        let s0 = symbols.get[0, Symbol]()
        let s1 = symbols.get[1, Symbol]()
        let s2 = symbols.get[2, Symbol]()
        let s3 = symbols.get[3, Symbol]()
        self.g = s0.g
        self.l = 4
        self.t = mobicc.new_tuple()
        mobicc.add_symbol(self.t, s0.s)
        mobicc.add_symbol(self.t, s1.s)
        mobicc.add_symbol(self.t, s2.s)
        mobicc.add_symbol(self.t, s3.s)

    # Request: @unique_value decorator for types
    # which acts like Value but only implements __moveinit__
    fn __moveinit__(inout self, owned existing: Self):
        self.t = existing.t ^
        self.l = existing.l
        self.g = existing.g

    fn __getitem__(self, pos: UInt32) -> Symbol:
        return Symbol(mobicc.get_symbol(self.t, pos), self.g)

    fn __len__(self) -> Int:
        return self.l
