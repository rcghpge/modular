# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct MOFDtype:
    """Mojo representation of engine's dtypes."""

    var dtype: Int32

    alias unknown: Int32 = 0

    alias int8: Int32 = 1
    alias int16: Int32 = 2
    alias int32: Int32 = 3
    alias int64: Int32 = 4

    alias uint8: Int32 = 10
    alias uint16: Int32 = 11
    alias uint32: Int32 = 12
    alias uint64: Int32 = 13

    alias float16: Int32 = 20
    alias float32: Int32 = 21
    alias float64: Int32 = 22

    alias bool: Int32 = 30

    fn __init__(dtype: DType) -> Self:
        if dtype == DType.int8:
            return Self {dtype: MOFDtype.int8}
        if dtype == DType.int16:
            return Self {dtype: MOFDtype.int16}
        if dtype == DType.int32:
            return Self {dtype: MOFDtype.int32}
        if dtype == DType.int64:
            return Self {dtype: MOFDtype.int64}

        if dtype == DType.uint8:
            return Self {dtype: MOFDtype.uint8}
        if dtype == DType.uint16:
            return Self {dtype: MOFDtype.uint16}
        if dtype == DType.uint32:
            return Self {dtype: MOFDtype.uint32}
        if dtype == DType.uint64:
            return Self {dtype: MOFDtype.uint64}

        if dtype == DType.float16:
            return Self {dtype: MOFDtype.float16}
        if dtype == DType.float32:
            return Self {dtype: MOFDtype.float32}
        if dtype == DType.float64:
            return Self {dtype: MOFDtype.float64}

        if dtype == DType.bool:
            return Self {dtype: MOFDtype.bool}

        return MOFDtype.unknown

    fn to_dtype(self) -> DType:
        if self == MOFDtype.int8:
            return DType.int8
        if self == MOFDtype.int16:
            return DType.int16
        if self == MOFDtype.int32:
            return DType.int32
        if self == MOFDtype.int64:
            return DType.int64

        if self == MOFDtype.uint8:
            return DType.uint8
        if self == MOFDtype.uint16:
            return DType.uint16
        if self == MOFDtype.uint32:
            return DType.uint32
        if self == MOFDtype.uint64:
            return DType.uint64

        if self == MOFDtype.float16:
            return DType.float16
        if self == MOFDtype.float32:
            return DType.float32
        if self == MOFDtype.float64:
            return DType.float64

        if self == MOFDtype.bool:
            return DType.bool

        return DType.invalid

    fn __eq__(self, other: MOFDtype) -> Bool:
        return self.dtype == other.dtype
