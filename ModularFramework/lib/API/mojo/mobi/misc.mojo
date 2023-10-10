# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .mobicc import ArityPtr, TuplePtr
from utils.list import VariadicList


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

    fn __init__(inout self, t: TuplePtr, l: Int):
        self.t = t
        self.l = l

    fn __len__(self) -> Int:
        return self.l
