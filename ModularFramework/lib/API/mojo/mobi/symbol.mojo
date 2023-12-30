# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .graph import Graph
from .mobicc import SymbolPtr, TuplePtr
from .motype import *
from .ops import *

from tensor import Tensor, TensorShape, TensorSpec


@value
@register_passable  # TODO: Use with Tuple shouldn't require reg-only.
struct Symbol(CollectionElement, Stringable):
    var s: SymbolPtr

    fn graph(self) -> Graph:
        return Graph(mobicc.get_symbol_graph(self.s))

    fn tensor_type(self) -> MOTensor:
        # TODO: Assert that this is an actual Tensor type, raise otherwise.
        let ranked = mobicc.is_tensor_ranked(self.s)
        let dims = mobicc.get_tensor_shape(self.s)
        let dtype = mobicc.get_tensor_dtype(self.s)
        return MOTensor(ElementType(dtype), dims, ranked)

    fn __str__(self) -> String:
        return mobicc.symbol_to_string(self.s)

    fn __getitem__(self, *spans: SliceSymbol) raises -> Symbol:
        return slice_(self, spans)

    fn __getitem__(self, span: slice) raises -> Symbol:
        if not span._has_end():
            raise "slice expects stop to be specified"

        let spec = self.tensor_type()
        if len(spec.dims) != 1:
            raise "__getitem__ with slice object on `Symbol` only supports 1D tensors"

        var g = self.graph()
        return g.op(
            "mo.slice",
            (
                self,
                g.i64_constant(g.m.i64(1), span.start),
                g.i64_constant(g.m.i64(1), span.end),
                g.i64_constant(g.m.i64(1), span.step),
            ),
            MOTensor(DType.int64, len(span)),
        )[0]

    fn print(self, label: StringRef = "debug_tensor") raises:
        var g = self.graph()
        let attrs = g.m.attrs(g.m.string_attr("label", label))
        _ = g.op("mo.debug.tensor.print", self, g.m.types(), attrs)

    fn transpose(self) raises -> Symbol:
        return transpose(self, -1, -2)

    fn transpose(self, dim1: Int, dim2: Int) raises -> Symbol:
        return transpose(self, dim1, dim2)

    fn __neg__(self) raises -> Symbol:
        var g = self.graph()
        return g.op("mo.negative", self, self.tensor_type())[0]

    fn __matmul__(self, rhs: Symbol) raises -> Symbol:
        return matmul(self, rhs)

    fn __add__(self, rhs: Symbol) raises -> Symbol:
        return add(self, rhs)

    fn __add__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return self + g.constant(rhs)

    fn __add__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.graph()
        return self + g.constant(rhs)

    fn __add__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return self + g.constant(rhs)

    fn __add__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return self + g.constant(rhs)

    fn __sub__(self, rhs: Symbol) raises -> Symbol:
        return sub(self, rhs)

    fn __sub__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return self - g.constant(rhs)

    fn __sub__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.graph()
        return self - g.constant(rhs)

    fn __sub__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return self - g.constant(rhs)

    fn __sub__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return self - g.constant(rhs)

    fn __mul__(self, rhs: Symbol) raises -> Symbol:
        return mul(self, rhs)

    fn __mul__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return self * g.constant(rhs)

    fn __mul__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.graph()
        return self * g.constant(rhs)

    fn __mul__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return self * g.constant(rhs)

    fn __mul__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return self * g.constant(rhs)

    fn __truediv__(self, rhs: Symbol) raises -> Symbol:
        return div(self, rhs)

    fn __truediv__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return self / g.constant(rhs)

    fn __truediv__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.graph()
        return self / g.constant(rhs)

    fn __truediv__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return self / g.constant(rhs)

    fn __truediv__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return self / g.constant(rhs)

    fn __pow__(self, rhs: Symbol) raises -> Symbol:
        return pow(self, rhs)

    fn __pow__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return self ** g.constant(rhs)

    fn __pow__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.graph()
        return self ** g.constant(rhs)

    fn __pow__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return self ** g.constant(rhs)

    fn __pow__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return self ** g.constant(rhs)

    fn __radd__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) + self

    fn __radd__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) + self

    fn __radd__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) + self

    fn __radd__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) + self

    fn __rsub__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) - self

    fn __rsub__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) - self

    fn __rsub__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) - self

    fn __rsub__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) - self

    fn __rmul__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) * self

    fn __rmul__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) * self

    fn __rmul__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) * self

    fn __rmul__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) * self

    fn __rtruediv__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) / self

    fn __rtruediv__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) / self

    fn __rtruediv__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) / self

    fn __rtruediv__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) / self

    fn __rpow__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) ** self

    fn __rpow__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) ** self

    fn __rpow__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) ** self

    fn __rpow__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) ** self


@value
struct Tup:
    var t: TuplePtr

    fn __init__(inout self):
        self.t = mobicc.new_tuple()

    fn __init__(inout self, *symbols: Symbol):
        self.__init__(symbols)

    fn __init__(inout self, symbols: VariadicList[Symbol]):
        self.t = mobicc.new_tuple()
        for symbol in symbols:
            mobicc.add_symbol(self.t, symbol.s)

    fn __init__(inout self, symbols: DynamicVector[Symbol]):
        self.t = mobicc.new_tuple()
        for i in range(len(symbols)):
            mobicc.add_symbol(self.t, symbols[i].s)

    fn __init__(inout self, symbol: Symbol):
        self.t = mobicc.new_tuple()
        mobicc.add_symbol(self.t, symbol.s)

    fn __init__(inout self, symbols: (Symbol, Symbol)):
        self.__init__(symbols.get[0, Symbol]().s, symbols.get[1, Symbol]().s)

    fn __init__(inout self, symbols: (Symbol, Symbol, Symbol)):
        self.__init__(
            symbols.get[0, Symbol]().s,
            symbols.get[1, Symbol]().s,
            symbols.get[2, Symbol]().s,
        )

    fn __init__(inout self, symbols: (Symbol, Symbol, Symbol, Symbol)):
        self.__init__(
            symbols.get[0, Symbol]().s,
            symbols.get[1, Symbol]().s,
            symbols.get[2, Symbol]().s,
            symbols.get[3, Symbol]().s,
        )

    fn __getitem__(self, pos: UInt32) -> Symbol:
        return Symbol(mobicc.get_symbol(self.t, pos))

    fn __len__(self) -> Int:
        return mobicc.tuple_size(self.t)
