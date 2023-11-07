# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .graph import Graph
from .mobicc import SymbolPtr


@value
struct Symbol:
    """Represents the result of an operation in `graph`, a graph input, or an
    MLIR block argument.
    """

    var s: SymbolPtr
    var g: Graph
