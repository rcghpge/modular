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


struct InferenceSession:
    var context: RuntimeContext
    var lib: DLHandle

    fn __init__(inout self, borrowed lib: DLHandle) raises:
        let config = RuntimeConfig(lib)
        self.context = RuntimeContext(config ^, lib)
        self.lib = lib

    fn __moveinit__(inout self, owned existing: Self):
        self.context = existing.context ^
        self.lib = existing.lib

    fn __enter__(owned self) -> Self:
        return self ^

    fn compile_model(self, model_path: StringRef) raises -> CompiledModel:
        """
        Compiles the model given model path.
        """
        return CompiledModel(self.context.borrow_ptr(), model_path, self.lib)

    fn load_model(self, model_path: StringRef) raises -> Model:
        """
        Compiles and initializes model given path.
        """
        let compiled_model = self.compile_model(model_path)
        let model = Model(
            self.context.borrow_ptr(), compiled_model.borrow_ptr(), self.lib
        )
        # Keep compiled model alive until model has been awaited.
        _ = compiled_model ^
        return model ^

    fn new_tensor_map(self) -> TensorMap:
        return TensorMap(self.context.borrow_ptr(), self.lib)
