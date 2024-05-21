# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from .device_tensor import DeviceTensor
from .tensor import Tensor
from .device import Device
from tensor import TensorSpec


struct AnyTensor:
    """A type erased tensor representation that is useful
    for situations where we need variadics of tensors."""

    var _data: DTypePointer[DType.invalid]
    var _spec: TensorSpec
    var _device: Device

    fn __init__(inout self, owned device_tensor: DeviceTensor) raises:
        self._device = device_tensor.get_device()
        self._spec = device_tensor.get_tensor_spec()
        self._data = device_tensor^._steal_ptr()

    fn __init__[
        type: DType, rank: Int
    ](inout self, owned tensor: Tensor[type, rank]) raises:
        self = Self(tensor^.get_device_tensor())

    fn get_rank(self) -> Int:
        """Gets rank of the tensor.

        Returns:
            Rank of the tensor.
        """
        return self._spec.rank()

    fn __del__(owned self):
        return self._device._free(self._data)
