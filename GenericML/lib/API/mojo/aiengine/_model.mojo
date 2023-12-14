# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from sys.intrinsics import _mlirtype_is_eq
from ._utils import *
from ._status import *
from ._context import *
from ._compilation import CCompiledModel
from ._tensormap import *
from ._tensor import *
from .session import InferenceSession


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


@value
struct Model:
    """Memory managed CModel."""

    var ctx: CRuntimeContext
    var ptr: CModel
    var lib: DLHandle
    var session: InferenceSession

    alias InitModelFnName = "M_initModel"
    alias ExecuteFnName = "M_executeModelSync"

    fn __moveinit__(inout self, owned existing: Self):
        self.ctx = existing.ctx
        self.ptr = exchange[CModel](
            existing.ptr, DTypePointer[DType.invalid].get_null()
        )
        self.lib = existing.lib
        self.session = existing.session ^

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
        return TensorMap(outputs, self.lib, self.session.copy())

    fn _execute_view[
        key_type: AnyRegType, value_type: AnyRegType
    ](
        self, inputs: VariadicList[Tuple[key_type, value_type]]
    ) raises -> TensorMap:
        let input_map = TensorMap(self.ctx, self.lib, self.session.copy())
        for i in range(len(inputs)):
            let pair = inputs[i]

            @parameter
            if _mlirtype_is_eq[key_type, StringRef]():

                @parameter
                if _mlirtype_is_eq[value_type, EngineTensorView]():
                    input_map.borrow(
                        pair.get[0, StringRef](),
                        pair.get[1, EngineTensorView](),
                    )
                elif _mlirtype_is_eq[value_type, EngineNumpyView]():
                    input_map.borrow(
                        pair.get[0, StringRef](), pair.get[1, EngineNumpyView]()
                    )

            elif _mlirtype_is_eq[key_type, StringLiteral]():

                @parameter
                if _mlirtype_is_eq[value_type, EngineTensorView]():
                    input_map.borrow(
                        pair.get[0, StringLiteral](),
                        pair.get[1, EngineTensorView](),
                    )
                elif _mlirtype_is_eq[value_type, EngineNumpyView]():
                    input_map.borrow(
                        pair.get[0, StringLiteral](),
                        pair.get[1, EngineNumpyView](),
                    )
        return self.execute(input_map)

    fn execute(
        self, *inputs: Tuple[StringRef, EngineTensorView]
    ) raises -> TensorMap:
        return self._execute_view[StringRef, EngineTensorView](inputs)

    fn execute(
        self, *inputs: Tuple[StringLiteral, EngineTensorView]
    ) raises -> TensorMap:
        return self._execute_view[StringLiteral, EngineTensorView](inputs)

    fn execute(
        self, *inputs: Tuple[StringLiteral, EngineNumpyView]
    ) raises -> TensorMap:
        return self._execute_view[StringLiteral, EngineNumpyView](inputs)

    fn __del__(owned self):
        self.ptr.free(self.lib)
        _ = self.session ^
