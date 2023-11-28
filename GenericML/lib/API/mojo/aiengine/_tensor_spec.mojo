# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from ._utils import *
from ._status import *
from tensor import TensorSpec
from ._dtypes import EngineDType
from utils.vector import DynamicVector
from .session import InferenceSession


@value
@register_passable("trivial")
struct CTensorSpec:
    """Mojo representation of Engine's TensorSpec pointer.
    This doesn't free the memory on destruction.
    """

    var ptr: DTypePointer[DType.invalid]

    alias FreeTensorSpecFnName = "M_freeTensorSpec"
    alias GetDimAtFnName = "M_getDimAt"
    alias GetRankFnName = "M_getRank"
    alias GetNameFnName = "M_getName"
    alias GetDTypeFnName = "M_getDtype"

    fn get_dim_at(self, idx: Int, lib: DLHandle) -> Int:
        return call_dylib_func[Int](lib, Self.GetDimAtFnName, self, idx)

    fn get_rank(self, lib: DLHandle) -> Int:
        return call_dylib_func[Int](lib, Self.GetRankFnName, self)

    fn get_name(self, lib: DLHandle) -> String:
        let name = call_dylib_func[CString](lib, Self.GetNameFnName, self)
        return name.__str__()

    fn get_dtype(self, lib: DLHandle) -> EngineDType:
        return call_dylib_func[EngineDType](lib, Self.GetDTypeFnName, self)

    fn free(self, borrowed lib: DLHandle):
        call_dylib_func(lib, Self.FreeTensorSpecFnName, self)


struct EngineTensorSpec:
    var ptr: CTensorSpec
    var lib: DLHandle
    var session: InferenceSession

    alias NewTensorSpecFnName = "M_newTensorSpec"

    fn __init__(
        inout self,
        ptr: CTensorSpec,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self.ptr = ptr
        self.lib = lib
        self.session = session ^

    fn __init__(
        inout self,
        name: StringRef,
        spec: TensorSpec,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        let dtype: EngineDType = spec.dtype()
        let rank = spec.rank()
        var shape = DynamicVector[Int64]()
        let name_str: CString = name.data
        for i in range(rank):
            shape.push_back(spec[i])
        self.ptr = call_dylib_func[CTensorSpec](
            lib, Self.NewTensorSpecFnName, shape.data, rank, dtype, name_str
        )
        _ = name
        _ = shape
        self.lib = lib
        self.session = session ^

    fn __getitem__(self, idx: Int) -> Int:
        return self.ptr.get_dim_at(idx, self.lib)

    fn get_as_tensor_spec(self) -> TensorSpec:
        var shape = DynamicVector[Int]()
        let rank = self.ptr.get_rank(self.lib)
        for i in range(rank):
            shape.push_back(self[i])
        let dtype = self.ptr.get_dtype(self.lib)
        let spec = TensorSpec(dtype.to_dtype(), shape)
        return spec

    fn get_name(self) -> String:
        return self.ptr.get_name(self.lib)

    fn __str__(self) -> String:
        var _repr: String = "{name="
        _repr += self.get_name()
        _repr += ", spec="
        _repr += self.get_as_tensor_spec().__str__()
        _repr += "}"
        return _repr

    fn borrow_ptr(self) -> CTensorSpec:
        return self.ptr

    fn __del__(owned self):
        self.ptr.free(self.lib)
        _ = self.session ^
