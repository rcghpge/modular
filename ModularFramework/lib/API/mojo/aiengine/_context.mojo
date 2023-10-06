# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from ._utils import *
from ._status import Status


@value
@register_passable("trivial")
struct CRuntimeConfig:
    var ptr: DTypePointer[DType.invalid]

    alias FreeRuntimeConfigFnName = "M_freeRuntimeConfig"

    fn free(self, borrowed lib: DLHandle):
        call_dylib_func(lib, Self.FreeRuntimeConfigFnName, self)


struct RuntimeConfig:
    var ptr: CRuntimeConfig
    var lib: DLHandle

    alias NewRuntimeConfigFnName = "M_newRuntimeConfig"

    fn __init__(inout self, borrowed lib: DLHandle):
        self.ptr = call_dylib_func[CRuntimeConfig](
            lib, Self.NewRuntimeConfigFnName
        )
        self.lib = lib

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = exchange[CRuntimeConfig](
            existing.ptr, DTypePointer[DType.invalid].get_null()
        )
        self.lib = existing.lib

    fn borrow_ptr(self) -> CRuntimeConfig:
        """
        Borrow the underlying C ptr.
        """
        return self.ptr

    fn __del__(owned self):
        self.ptr.free(self.lib)


@value
@register_passable("trivial")
struct CRuntimeContext:
    var ptr: DTypePointer[DType.invalid]

    alias FreeRuntimeContextFnName = "M_freeRuntimeContext"

    fn free(self, borrowed lib: DLHandle):
        call_dylib_func(lib, Self.FreeRuntimeContextFnName, self)


struct RuntimeContext:
    var ptr: CRuntimeContext
    var lib: DLHandle

    alias NewRuntimeContextFnName = "M_newRuntimeContext"

    fn __init__(
        inout self, owned config: RuntimeConfig, borrowed lib: DLHandle
    ) raises:
        let status = Status(lib)
        self.ptr = call_dylib_func[CRuntimeContext](
            lib,
            Self.NewRuntimeContextFnName,
            config.borrow_ptr(),
            status.borrow_ptr(),
        )
        if status:
            raise Error(status.__str__())
        _ = config ^
        self.lib = lib

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = exchange[CRuntimeContext](
            existing.ptr, DTypePointer[DType.invalid].get_null()
        )
        self.lib = existing.lib

    fn borrow_ptr(self) -> CRuntimeContext:
        return self.ptr

    fn __del__(owned self):
        self.ptr.free(self.lib)
