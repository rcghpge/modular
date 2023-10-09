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
from ._compilation import CCompiledModel
from ._tensormap import *
from ._tensor import *
from utils.list import VariadicList


@value
@register_passable("trivial")
struct CModel:
    """Mojo representation of Engine's AsyncModel pointer.
    Useful for C inter-op.
    """

    var ptr: DTypePointer[DType.invalid]

    alias FreeModelFnName = "M_freeModel"
    alias WaitForModelFnName = "M_waitForModel"

    fn await_model(self, lib: DLHandle) raises:
        let status = Status(lib)
        call_dylib_func(
            lib, Self.WaitForModelFnName, self.ptr, status.borrow_ptr()
        )
        if status:
            raise status.__str__()

    fn free(self, borrowed lib: DLHandle):
        call_dylib_func(lib, Self.FreeModelFnName, self)


struct Model:
    """Memory managed CModel."""

    var ctx: CRuntimeContext
    var ptr: CModel
    var lib: DLHandle

    alias InitModelFnName = "M_initModel"
    alias ExecuteFnName = "M_executeModelSync"

    fn __init__(
        inout self,
        context: CRuntimeContext,
        source: CCompiledModel,
        lib: DLHandle,
    ) raises:
        self.ctx = context
        let status = Status(lib)
        self.ptr = call_dylib_func[CModel](
            lib, Self.InitModelFnName, context, source, status.borrow_ptr()
        )
        if status:
            raise status.__str__()

        self.ptr.await_model(lib)
        self.lib = lib

    fn __moveinit__(inout self, owned existing: Self):
        self.ctx = existing.ctx
        self.ptr = exchange[CModel](
            existing.ptr, DTypePointer[DType.invalid].get_null()
        )
        self.lib = existing.lib

    fn execute(self, inputs: TensorMap) raises -> TensorMap:
        let status = Status(self.lib)
        let outputs = call_dylib_func[CTensorMap](
            self.lib,
            Self.ExecuteFnName,
            self.ctx,
            self.ptr,
            inputs.borrow_ptr(),
            status.borrow_ptr(),
        )
        if status:
            raise status.__str__()
        return TensorMap(outputs, self.lib)

    fn execute(
        self, *inputs: Tuple[StringRef, EngineTensorView]
    ) raises -> TensorMap:
        let input_map = TensorMap(self.ctx, self.lib)
        let inputs_list = VariadicList[Tuple[StringRef, EngineTensorView]](
            inputs
        )
        for i in range(inputs_list.__len__()):
            let pair = inputs_list[i]
            input_map.borrow(
                pair.get[0, StringRef](), pair.get[1, EngineTensorView]()
            )
        return self.execute(input_map)

    fn execute(
        self, *inputs: Tuple[StringLiteral, EngineTensorView]
    ) raises -> TensorMap:
        let input_map = TensorMap(self.ctx, self.lib)
        let inputs_list = VariadicList[Tuple[StringLiteral, EngineTensorView]](
            inputs
        )
        for i in range(inputs_list.__len__()):
            let pair = inputs_list[i]
            input_map.borrow(
                pair.get[0, StringLiteral](), pair.get[1, EngineTensorView]()
            )
        return self.execute(input_map)

    fn __del__(owned self):
        self.ptr.free(self.lib)
