# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .mobicc import AttrMapPtr


struct AttrMap:
    var m: AttrMapPtr
    var l: Int

    fn __init__(inout self, m: AttrMapPtr, l: Int):
        self.m = m
        self.l = l

    fn __len__(self) -> Int:
        return self.l
