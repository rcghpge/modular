# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from ._context import *
from ._status import *
from sys.ffi import DLHandle
from ._compilation import CompiledModel
from ._model import Model
from ._tensormap import *
from ._engine import _EngineImpl


struct InferenceSession:
    var engine: _EngineImpl
    var context: RuntimeContext

    fn __init__(inout self) raises:
        self.engine = _EngineImpl()
        let config = RuntimeConfig(self.engine.lib)
        self.context = RuntimeContext(config ^, self.engine.lib)

    fn __moveinit__(inout self, owned existing: Self):
        self.engine = existing.engine ^
        self.context = existing.context ^

    fn __enter__(owned self) -> Self:
        return self ^

    fn compile_model(self, model_path: StringRef) raises -> CompiledModel:
        """
        Compiles the model given model path.
        """
        return CompiledModel(
            self.context.borrow_ptr(), model_path, self.engine.lib
        )

    fn load_model(self, model_path: StringRef) raises -> Model:
        """
        Compiles and initializes model given path.
        """
        let compiled_model = self.compile_model(model_path)
        let model = Model(
            self.context.borrow_ptr(),
            compiled_model.borrow_ptr(),
            self.engine.lib,
        )
        # Keep compiled model alive until model has been awaited.
        _ = compiled_model ^
        return model ^

    fn get_as_engine_tensor_spec(
        self, name: StringRef, spec: TensorSpec
    ) -> EngineTensorSpec:
        return EngineTensorSpec(name, spec, self.engine.lib)

    fn new_tensor_map(self) -> TensorMap:
        return TensorMap(self.context.borrow_ptr(), self.engine.lib)

    fn get_status(self) -> Status:
        """Creates a Status object."""
        return Status(self.engine.lib)
