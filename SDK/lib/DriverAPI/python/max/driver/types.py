# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from enum import IntEnum


class dtype(IntEnum):
    bool = 0
    int8 = 4
    int16 = 5
    int32 = 6
    int64 = 7
    uint8 = 15
    uint16 = 16
    uint32 = 17
    uint64 = 18
    float16 = 25
    float32 = 26
    float64 = 27
    bfloat16 = 33
    unknown = -1
