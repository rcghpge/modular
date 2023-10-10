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
from tensor import Tensor, TensorShape
from ._tensor import *
from ._tensor_spec import *
from ._context import *
from .session import InferenceSession


@value
@register_passable("trivial")
struct CTensorMap:
    """Represents AsyncTensorMap ptr from Engine."""

    var ptr: DTypePointer[DType.invalid]

    alias FreeAsyncTensorMapFnName = "M_freeAsyncTensorMap"
    alias BorrowTensorIntoFnName = "M_borrowTensorInto"
    alias GetTensorByNameFromFnName = "M_getTensorByNameFrom"
    alias GetTensorMapSizeFnName = "M_getTensorMapSize"

    fn get_tensor_by_name(self, name: CString, lib: DLHandle) raises -> CTensor:
        let status = Status(lib)
        let tensor = call_dylib_func[CTensor](
            lib, Self.GetTensorByNameFromFnName, self, name, status.borrow_ptr()
        )
        if status:
            raise status.__str__()
        return tensor

    fn borrow_tensor_by_name(
        self,
        ptr: DTypePointer[DType.invalid],
        spec: EngineTensorSpec,
        lib: DLHandle,
    ) raises:
        let status = Status(lib)
        call_dylib_func(
            lib,
            Self.BorrowTensorIntoFnName,
            self,
            ptr,
            spec.borrow_ptr(),
            status.borrow_ptr(),
        )
        if status:
            raise status.__str__()

    fn size(self, borrowed lib: DLHandle) raises -> Int:
        let status = Status(lib)
        let size = call_dylib_func[Int](
            lib, Self.GetTensorMapSizeFnName, self, status.borrow_ptr()
        )
        if status:
            raise status.__str__()
        return size

    fn free(self, borrowed lib: DLHandle):
        """
        Free the status ptr.
        """
        call_dylib_func(lib, Self.FreeAsyncTensorMapFnName, self)


struct TensorMap:
    var ptr: CTensorMap
    var lib: DLHandle
    var session: InferenceSession

    alias NewTensorMapFnName = "M_newAsyncTensorMap"

    fn __init__(
        inout self,
        ctx: CRuntimeContext,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self.ptr = call_dylib_func[CTensorMap](
            lib, Self.NewTensorMapFnName, ctx
        )
        self.lib = lib
        self.session = session ^

    fn __init__(
        inout self,
        ptr: CTensorMap,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self.ptr = ptr
        self.lib = lib
        self.session = session ^

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = exchange[CTensorMap](
            existing.ptr, DTypePointer[DType.invalid]()
        )
        self.lib = existing.lib
        self.session = existing.session ^

    fn borrow[type: DType](self, key: StringRef, value: Tensor[type]) raises:
        let spec = EngineTensorSpec(
            key, value.spec(), self.lib, self.session.copy()
        )
        self.ptr.borrow_tensor_by_name(
            bitcast[DType.invalid](value.data()), spec, self.lib
        )

    fn borrow(self, key: StringRef, value: EngineTensorView) raises:
        let spec = EngineTensorSpec(
            key, value.spec(), self.lib, self.session.copy()
        )
        self.ptr.borrow_tensor_by_name(value.data(), spec, self.lib)

    fn get[type: DType](self, key: StringRef) raises -> Tensor[type]:
        let tensor_ptr = self.ptr.get_tensor_by_name(key.data, self.lib)
        let mof_tensor = EngineTensor(tensor_ptr, self.lib, self.session.copy())
        let tensor = mof_tensor.tensor[type]()
        return tensor ^

    fn __len__(self) raises -> Int:
        return self.ptr.size(self.lib)

    fn borrow_ptr(self) -> CTensorMap:
        return self.ptr

    fn __del__(owned self):
        self.ptr.free(self.lib)
        _ = self.session ^
