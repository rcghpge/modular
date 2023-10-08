# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from ._utils import *
from ._dtypes import *
from ._tensor_spec import *
from memory.unsafe import bitcast
from tensor import Tensor


@value
@register_passable("trivial")
struct CTensor:
    """Represents AsyncTensor ptr from Engine."""

    var ptr: DTypePointer[DType.invalid]

    alias GetTensorNumElementsFnName = "M_getTensorNumElements"
    alias GetTensorDTypeFnName = "M_getTensorType"
    alias GetTensorDataFnName = "M_getTensorData"
    alias GetTensorSpecFnName = "M_getTensorSpec"
    alias FreeTensorFnName = "M_freeTensor"

    fn size(self, borrowed lib: DLHandle) -> Int:
        return call_dylib_func[Int](lib, Self.GetTensorNumElementsFnName, self)

    fn dtype(self, borrowed lib: DLHandle) -> EngineDType:
        return call_dylib_func[EngineDType](
            lib, Self.GetTensorDTypeFnName, self
        )

    fn data(self, borrowed lib: DLHandle) -> DTypePointer[DType.invalid]:
        return call_dylib_func[DTypePointer[DType.invalid]](
            lib, Self.GetTensorDataFnName, self
        )

    fn get_tensor_spec(self, borrowed lib: DLHandle) -> EngineTensorSpec:
        let spec = call_dylib_func[CTensorSpec](
            lib, Self.GetTensorSpecFnName, self
        )
        return EngineTensorSpec(spec, lib)

    fn free(self, borrowed lib: DLHandle):
        """
        Free the status ptr.
        """
        call_dylib_func(lib, Self.FreeTensorFnName, self)


struct EngineTensor:
    var ptr: CTensor
    var lib: DLHandle

    fn __init__(inout self, ptr: CTensor, lib: DLHandle):
        self.ptr = ptr
        self.lib = lib

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = exchange[CTensor](
            existing.ptr, DTypePointer[DType.invalid]()
        )
        self.lib = existing.lib

    fn __len__(self) -> Int:
        return self.ptr.size(self.lib)

    fn data(self) -> DTypePointer[DType.invalid]:
        return self.ptr.data(self.lib)

    fn data[type: DType](self) raises -> DTypePointer[type]:
        let ptr = self.data()
        return bitcast[type](ptr)

    fn dtype(self) -> DType:
        return self.ptr.dtype(self.lib).to_dtype()

    fn spec(self) -> TensorSpec:
        return self.ptr.get_tensor_spec(self.lib).get_as_tensor_spec()

    fn buffer[type: DType](self) raises -> Buffer[Dim(), type]:
        return Buffer[Dim(), type](self.data[type](), self.__len__())

    fn buffer(self) -> Buffer[Dim(), DType.invalid]:
        return Buffer[Dim(), DType.invalid](
            self.data(), self.__len__() * self.dtype().sizeof()
        )

    fn tensor[type: DType](self) raises -> Tensor[type]:
        let tensor = Tensor[type](self.spec())
        memcpy(
            tensor.data(),
            self.data[type](),
            self.__len__(),
        )
        return tensor ^

    fn __del__(owned self):
        self.ptr.free(self.lib)
