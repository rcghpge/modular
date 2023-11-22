# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .graph import Graph
from .mobicc import SymbolPtr
from .type import Type
import mobi.functional as G

from tensor import Tensor, TensorShape, TensorSpec


@value
@register_passable
struct Symbol:
    """Represents the result of an operation in `graph`, a graph input, or an
    MLIR block argument.
    """

    var s: SymbolPtr
    var g: Graph

    fn __init__(s: SymbolPtr, g: Graph) -> Self:
        return Self {s: s, g: g}

    fn type(self) -> Type:
        """Returns the type of this symbol."""
        return Type(mobicc.typeof(self.s))

    fn __str__(self) raises -> String:
        let spec = self.type().unwrap_tensor().spec()
        var shape = String("[")
        for i in range(spec.rank()):
            if i > 0:
                shape += ", "
            shape += "?" if spec[i] < 0 else String(spec[i])
        shape += "]"
        let dtype = String(spec.dtype().__str__())
        return "Symbol(mo.tensor<" + shape + ", " + dtype + ">)"

    fn __getitem__(self, *spans: G.SliceSymbol) raises -> Symbol:
        return G.slice_(self, spans)

    fn __getitem__(self, span: slice) raises -> Symbol:
        # TODO: Since `mo.slice` clamps out of range start/stop operands, it
        # should be safe to leave unspecified `end` as `_int_max_value()`.
        # However, since this will produce a dynamic dim shape that is prone to
        # producing downstream dynamic rank, enforce statically known `end`.
        # var g = self.g
        # return G.slice_(
        #     self,
        #     G.SliceSymbol(
        #         g.constant(span.start),
        #         g.constant(span.end),
        #         g.constant(span.step),
        #     ),
        # )

        if not span._has_end():
            raise "slice expects stop to be specified"

        let spec = self.type().unwrap_tensor().spec()
        if spec.rank() != 1:
            raise "__getitem__ with slice object on `Symbol` only supports 1D tensors"

        # Allocated pointers are moved into `Tensor`, which frees them.
        let start_ptr = DTypePointer[DType.int64].alloc(1)
        start_ptr.store(0, span.start)
        let end_ptr = DTypePointer[DType.int64].alloc(1)
        end_ptr.store(0, span.end)
        let step_ptr = DTypePointer[DType.int64].alloc(1)
        step_ptr.store(0, span.step)

        var g = self.g
        return g.op(
            "mo.slice",
            (
                self,
                g.constant(Tensor(start_ptr, TensorShape(1))),
                g.constant(Tensor(end_ptr, TensorShape(1))),
                g.constant(Tensor(step_ptr, TensorShape(1))),
            ),
            TensorSpec(DType.int64, len(span)),
        )[0]

    fn print(self, label: StringRef = "debug_tensor") raises:
        var g = self.g
        let attrs = g.m.attrs(g.m.string_attr("label", label))
        _ = g.op("mo.debug.tensor.print", self, g.m.types(), attrs)

    fn transpose(self) raises -> Symbol:
        return G.transpose(self, -1, -2)

    fn transpose(self, dim1: Int, dim2: Int) raises -> Symbol:
        return G.transpose(self, dim1, dim2)

    fn __matmul__(self, rhs: Symbol) raises -> Symbol:
        return G.matmul(self, rhs)

    fn __add__(self, rhs: Symbol) raises -> Symbol:
        return G.add(self, rhs)

    fn __add__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.g
        return self + g.constant(rhs)

    fn __add__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.g
        return self + g.constant(rhs)

    fn __add__(self, rhs: Int) raises -> Symbol:
        var g = self.g
        return self + g.constant(rhs)

    fn __add__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.g
        return self + g.constant(rhs)

    fn __sub__(self, rhs: Symbol) raises -> Symbol:
        return G.sub(self, rhs)

    fn __sub__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.g
        return self - g.constant(rhs)

    fn __sub__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.g
        return self - g.constant(rhs)

    fn __sub__(self, rhs: Int) raises -> Symbol:
        var g = self.g
        return self - g.constant(rhs)

    fn __sub__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.g
        return self - g.constant(rhs)

    fn __mul__(self, rhs: Symbol) raises -> Symbol:
        return G.mul(self, rhs)

    fn __mul__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.g
        return self * g.constant(rhs)

    fn __mul__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.g
        return self * g.constant(rhs)

    fn __mul__(self, rhs: Int) raises -> Symbol:
        var g = self.g
        return self * g.constant(rhs)

    fn __mul__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.g
        return self * g.constant(rhs)

    fn __truediv__(self, rhs: Symbol) raises -> Symbol:
        return G.div(self, rhs)

    fn __truediv__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.g
        return self / g.constant(rhs)

    fn __truediv__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.g
        return self / g.constant(rhs)

    fn __truediv__(self, rhs: Int) raises -> Symbol:
        var g = self.g
        return self / g.constant(rhs)

    fn __truediv__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.g
        return self / g.constant(rhs)

    fn __pow__(self, rhs: Symbol) raises -> Symbol:
        return G.pow(self, rhs)

    fn __pow__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.g
        return self ** g.constant(rhs)

    fn __pow__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.g
        return self ** g.constant(rhs)

    fn __pow__(self, rhs: Int) raises -> Symbol:
        var g = self.g
        return self ** g.constant(rhs)

    fn __pow__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.g
        return self ** g.constant(rhs)

    fn __radd__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) + self

    fn __radd__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) + self

    fn __radd__(self, rhs: Int) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) + self

    fn __radd__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) + self

    fn __rsub__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) - self

    fn __rsub__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) - self

    fn __rsub__(self, rhs: Int) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) - self

    fn __rsub__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) - self

    fn __rmul__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) * self

    fn __rmul__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) * self

    fn __rmul__(self, rhs: Int) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) * self

    fn __rmul__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) * self

    fn __rtruediv__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) / self

    fn __rtruediv__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) / self

    fn __rtruediv__(self, rhs: Int) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) / self

    fn __rtruediv__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) / self

    fn __rpow__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) ** self

    fn __rpow__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) ** self

    fn __rpow__(self, rhs: Int) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) ** self

    fn __rpow__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.g
        return g.constant(rhs) ** self
