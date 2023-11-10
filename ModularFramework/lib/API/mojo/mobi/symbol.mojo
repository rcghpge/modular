# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .graph import Graph
from .mobicc import SymbolPtr
from .type import Type


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
