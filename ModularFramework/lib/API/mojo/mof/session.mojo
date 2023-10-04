# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from ._context import *
from ._status import *
from sys.ffi import DLHandle
from ._compilation import CompiledModel


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
