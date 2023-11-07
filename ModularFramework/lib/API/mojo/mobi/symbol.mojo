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

    fn type(self) -> Type:
        """Returns the type of this symbol."""
        return Type(mobicc.typeof(self.s))
