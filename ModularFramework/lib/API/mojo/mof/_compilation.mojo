# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from ._utils import *
from ._status import *
from ._context import *


@value
@register_passable("trivial")
struct CCompileConfig:
    """Mojo representation of Engine's CompileConfig pointer.
    This doesn't free the memory on destruction. For memory managed
    option see CompileConfig.
    """

    var ptr: DTypePointer[DType.invalid]

    alias FreeCompileConfigFnName = "M_freeCompileConfig"
    alias SetModelPathFnName = "M_setModelPath"

    fn set_model_path(self, path: StringRef, borrowed lib: DLHandle):
        """Sets the path of model to compile."""
        call_dylib_func(lib, Self.SetModelPathFnName, self, path.data)

    fn free(self, borrowed lib: DLHandle):
        call_dylib_func(lib, Self.FreeCompileConfigFnName, self)


struct CompileConfig:
    """Memory managed version of Engine's Compile Config."""

    var ptr: CCompileConfig
    var lib: DLHandle

    alias NewCompileConfigFnName = "M_newCompileConfig"

    fn __init__(inout self, borrowed lib: DLHandle):
        self.ptr = call_dylib_func[CCompileConfig](
            lib, Self.NewCompileConfigFnName
        )
        self.lib = lib

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = exchange[CCompileConfig](
            existing.ptr, DTypePointer[DType.invalid].get_null()
        )
        self.lib = existing.lib

    fn set_model_path(self, path: StringRef):
        """Sets the path of model to compile."""
        self.ptr.set_model_path(path, self.lib)

    fn borrow_ptr(self) -> CCompileConfig:
        return self.ptr

    fn __del__(owned self):
        self.ptr.free(self.lib)


@value
@register_passable("trivial")
struct CCompiledModel:
    """Mojo representation of Engine's AsyncCompiledModel pointer.
    Useful for C inter-op.
    """

    var ptr: DTypePointer[DType.invalid]

    alias FreeCompiledModelFnName = "M_freeCompiledModel"
    alias GetNumInputsFnName = "M_getNumModelInputs"

    fn num_model_inputs(self, borrowed lib: DLHandle) raises -> Int:
        """Gets the number of inputs of the model."""

        let status = Status(lib)
        let num_inputs = call_dylib_func[Int](
            lib, Self.GetNumInputsFnName, self, status.ptr
        )
        if status:
            raise Error(status.__str__())
        return num_inputs

    fn free(self, borrowed lib: DLHandle):
        call_dylib_func(lib, Self.FreeCompiledModelFnName, self)


struct CompiledModel:
    """Memory managed CompiledModel pointer."""

    var ptr: CCompiledModel
    var lib: DLHandle

    alias CompileModelFnName = "M_compileModelSync"

    fn __init__(
        inout self,
        borrowed ctx: CRuntimeContext,
        path: StringRef,
        borrowed lib: DLHandle,
    ) raises:
        let config = CompileConfig(lib)
        config.set_model_path(path)
        let status = Status(lib)
        self.ptr = call_dylib_func[CCompiledModel](
            lib,
            Self.CompileModelFnName,
            ctx,
            config.borrow_ptr(),
            status.borrow_ptr(),
        )
        _ = config ^
        if status:
            raise Error(status.__str__())
        self.lib = lib

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = exchange[CCompiledModel](
            existing.ptr, DTypePointer[DType.invalid].get_null()
        )
        self.lib = existing.lib

    fn num_model_inputs(self) raises -> Int:
        """Gets the number of inputs of the model."""

        return self.ptr.num_model_inputs(self.lib)

    fn __del__(owned self):
        self.ptr.free(self.lib)
