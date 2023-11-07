# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .graph import Graph
from .mobicc import SymbolPtr
from .type import Type


@value
struct Symbol:
    """Represents the result of an operation in `graph`, a graph input, or an
    MLIR block argument.
    """

    var s: SymbolPtr
    var g: Graph

    fn builder(self) -> Builder:
        """Returns a `builder` handle to the underlying shared library."""
        return self.g.m.builder

    fn type(self) -> Type:
        """Returns the type of this symbol."""
        return Type(self.builder().typeof(self.s), self.builder())
