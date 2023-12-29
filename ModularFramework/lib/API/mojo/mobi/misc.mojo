# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections.vector import DynamicVector

from .graph import Graph
from .mobicc import AttrMapPtr, ArityPtr
from .type import Type

# TODO: Move these to type/attr.


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
