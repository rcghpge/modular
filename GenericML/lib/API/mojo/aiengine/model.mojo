# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from ._model_impl import CModel
from ._context import CRuntimeContext
from sys.ffi import DLHandle
from ._utils import *
from ._status import Status
from .tensor_map import *
from ._tensor_impl import *
from sys.intrinsics import _mlirtype_is_eq


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
