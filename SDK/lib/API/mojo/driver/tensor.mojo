# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from .device import Device


@value
struct _StaticTensorSpec[type: DType, rank: Int]:
    var shape: StaticIntTuple[rank]

    fn get_rank(self) -> Int:
        return rank

    fn dtype(self) -> DType:
        return type


struct Tensor[type: DType, rank: Int]:
    var _ptr: DTypePointer[type]
    var _spec: _StaticTensorSpec[type, rank]
    var _device: Device

    fn __init__(
        inout self,
        ptr: DTypePointer[type],
        shape: StaticIntTuple[rank],
        device: Device,
    ):
        self._ptr = ptr
        self._spec = _StaticTensorSpec[type](shape)
        self._device = device

    fn __moveinit__(inout self, owned existing: Self):
        self._ptr = existing._ptr
        self._spec = existing._spec
        self._device = existing._device

    fn get_rank(self) -> Int:
        return self._spec.get_rank()

    fn get_dtype(self) -> DType:
        return self._spec.dtype()

    fn __del__(owned self):
        self._device._free(self._ptr.bitcast[DType.invalid]())
