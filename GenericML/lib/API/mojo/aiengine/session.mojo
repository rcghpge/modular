# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from ._context import *
from ._status import *
from sys.ffi import DLHandle
from ._compilation import CompiledModel, CompileConfig
from ._model import Model
from ._tensormap import *
from ._engine import _EngineImpl
from memory.unsafe import Pointer
from os.atomic import Atomic
from utils._optional import Optional


struct _InferenceSessionImpl:
    var engine: _EngineImpl
    var context: RuntimeContext
    var ref_count: Atomic[DType.int64]

    fn __init__(inout self) raises:
        self.engine = _EngineImpl()
        let config = RuntimeConfig(self.engine.lib)
        self.context = RuntimeContext(config ^, self.engine.lib)
        self.ref_count = 1

    fn compile_model(
        self,
        model_path: StringRef,
        user_defined_ops_path: Optional[StringRef],
        owned session: InferenceSession,
    ) raises -> CompiledModel:
        """
        Compiles the model given model path.

        Raises: an Error if the path to the model file does
        not exist on the filesystem.
        """
        let config = CompileConfig(self.engine.lib)
        if user_defined_ops_path:
            config.set_replace_ops_path(user_defined_ops_path.value())
        config.set_model_path(model_path)

        return CompiledModel(
            self.context.borrow_ptr(),
            self.engine.lib,
            config ^,
            session ^,
        )

    fn load_model(
        self,
        model_path: StringRef,
        user_defined_ops_path: Optional[StringRef],
        owned session: InferenceSession,
    ) raises -> Model:
        """
        Compiles and initializes model given path.
        """
        let compiled_model = self.compile_model(
            model_path, user_defined_ops_path, session.copy()
        )
        let model = Model(
            self.context.borrow_ptr(),
            compiled_model.borrow_ptr(),
            self.engine.lib,
            session ^,
        )
        # Keep compiled model alive until model has been awaited.
        _ = compiled_model ^
        return model ^

    fn get_as_engine_tensor_spec(
        self,
        name: StringRef,
        spec: TensorSpec,
        owned session: InferenceSession,
    ) -> EngineTensorSpec:
        return EngineTensorSpec(name, spec, self.engine.lib, session ^)

    fn new_tensor_map(self, owned session: InferenceSession) -> TensorMap:
        return TensorMap(self.context.borrow_ptr(), self.engine.lib, session ^)


@register_passable
struct InferenceSession:
    var ptr: Pointer[_InferenceSessionImpl]

    fn __init__() raises -> Self:
        let ptr = Pointer[_InferenceSessionImpl].alloc(1)
        __get_address_as_uninit_lvalue(ptr.address).__init__()
        return Self {ptr: ptr}

    fn copy(self) -> Self:
        _ = __get_address_as_lvalue(self.ptr.address).ref_count.fetch_add(1)
        return Self {ptr: self.ptr}

    fn compile_model(
        self,
        model_path: StringRef,
        user_defined_ops_path: Optional[StringRef] = None,
    ) raises -> CompiledModel:
        """
        If provided, Replaces Modular ops with user-defined ops using the provided
        path-to-mojopkg and then compiles the model using the provided
        model path.
        """
        return __get_address_as_lvalue(self.ptr.address).compile_model(
            model_path, user_defined_ops_path, self.copy()
        )

    fn load_model(self, path: Path) raises -> Model:
        let model = self.load_model(path.path._strref_dangerous())
        path.path._strref_keepalive()
        return model ^

    fn load_model(
        self,
        model_path: StringRef,
        user_defined_ops_path: Optional[StringRef] = None,
    ) raises -> Model:
        """
        Replaces Modular ops with user-defined ops using the provided
        path-to-mojopkg and then compiles and initializes the model
        using the provided model path.
        """
        return __get_address_as_lvalue(self.ptr.address).load_model(
            model_path, user_defined_ops_path, self.copy()
        )

    fn get_as_engine_tensor_spec(
        self, name: StringRef, spec: TensorSpec
    ) -> EngineTensorSpec:
        return __get_address_as_lvalue(
            self.ptr.address
        ).get_as_engine_tensor_spec(name, spec, self.copy())

    fn new_tensor_map(self) -> TensorMap:
        return __get_address_as_lvalue(self.ptr.address).new_tensor_map(
            self.copy()
        )

    fn __del__(owned self):
        if (
            __get_address_as_lvalue(self.ptr.address).ref_count.fetch_sub(1)
            != 1
        ):
            # There are others holding reference to this session. Keep the
            # session alive and let other reference holders deal with
            # managing it.
            return

        # Now we know that there is only active reference to this session
        # and that is held by us. So go ahead and call destructor on it.
        _ = __get_address_as_owned_value(self.ptr.address)
        self.ptr.free()
