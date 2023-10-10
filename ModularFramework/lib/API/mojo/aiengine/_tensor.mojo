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
from memory.unsafe import bitcast, Pointer
from tensor import Tensor
from .session import InferenceSession


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

    fn get_tensor_spec(
        self, borrowed lib: DLHandle, owned session: InferenceSession
    ) -> EngineTensorSpec:
        let spec = call_dylib_func[CTensorSpec](
            lib, Self.GetTensorSpecFnName, self
        )
        return EngineTensorSpec(spec, lib, session ^)

    fn free(self, borrowed lib: DLHandle):
        """
        Free the status ptr.
        """
        call_dylib_func(lib, Self.FreeTensorFnName, self)


struct EngineTensor:
    var ptr: CTensor
    var lib: DLHandle
    var session: InferenceSession

    fn __init__(
        inout self,
        ptr: CTensor,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self.ptr = ptr
        self.lib = lib
        self.session = session ^

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = exchange[CTensor](
            existing.ptr, DTypePointer[DType.invalid]()
        )
        self.lib = existing.lib
        self.session = existing.session ^

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
        return self.ptr.get_tensor_spec(
            self.lib, self.session.copy()
        ).get_as_tensor_spec()

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


@register_passable
struct EngineTensorView:
    """A non-owning register_passable view of a tensor
    that does runtime type checking.

    CAUTION: Make sure the source tensor outlives the view.
    """

    var ptr: Pointer[Tensor[DType.invalid]]
    var data_ptr: DTypePointer[DType.invalid]
    var dtype: DType

    fn __init__[type: DType](inout tensor: Tensor[type]) -> Self:
        return Self {
            ptr: bitcast[Tensor[DType.invalid]](
                Pointer[Tensor[type]].address_of(tensor)
            ),
            data_ptr: bitcast[DType.invalid](tensor.data()),
            dtype: type,
        }

    fn data[type: DType](self) raises -> DTypePointer[type]:
        if type != self.dtype:
            raise String("Expected type: ") + self.dtype.__str__()
        return bitcast[type](self.data_ptr)

    fn data(self) -> DTypePointer[DType.invalid]:
        return self.data_ptr

    fn spec(self) raises -> TensorSpec:
        @always_inline
        @parameter
        fn get_spec[ty: DType]() -> TensorSpec:
            return self._get_value[ty]().spec()

        if self.dtype.is_int8():
            return get_spec[DType.int8]()
        if self.dtype.is_int16():
            return get_spec[DType.int16]()
        if self.dtype.is_int32():
            return get_spec[DType.int32]()
        if self.dtype.is_int64():
            return get_spec[DType.int64]()

        if self.dtype.is_uint8():
            return get_spec[DType.uint8]()
        if self.dtype.is_uint16():
            return get_spec[DType.uint16]()
        if self.dtype.is_uint32():
            return get_spec[DType.uint32]()
        if self.dtype.is_uint64():
            return get_spec[DType.uint64]()

        if self.dtype.is_float16():
            return get_spec[DType.float16]()
        if self.dtype.is_float32():
            return get_spec[DType.float32]()
        if self.dtype.is_float64():
            return get_spec[DType.float64]()
        if self.dtype.is_bool():
            return get_spec[DType.bool]()

        raise String("Expected type: ") + self.dtype.__str__()

    @always_inline("nodebug")
    fn _get_value[type: DType](self) -> Tensor[type]:
        return __get_address_as_lvalue(bitcast[Tensor[type]](self.ptr).address)
