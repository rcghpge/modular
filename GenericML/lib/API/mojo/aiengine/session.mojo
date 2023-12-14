# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from ._context import *
from ._status import *
from sys.ffi import DLHandle
from ._compilation import *
from ._model import *
from ._tensormap import *
from ._engine import _EngineImpl, get_engine_path
from memory.unsafe import Pointer
from os.atomic import Atomic
from utils._optional import Optional


struct _InferenceSessionImpl:
    var engine: _EngineImpl
    var context: RuntimeContext
    var ref_count: Atomic[DType.int64]

    fn __init__(
        inout self,
        lib_path: StringRef,
        device: StringRef = "cpu",
        num_threads: Optional[Int] = None,
    ):
        self.engine = _EngineImpl(lib_path)
        let config = RuntimeConfig(self.engine.lib, device, num_threads)
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
        let context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to compile model"
        let config = CompileConfig(self.engine.lib)
        if user_defined_ops_path:
            config.set_replace_ops_path(user_defined_ops_path.value())
        config.set_model_path(model_path)

        let status = Status(self.engine.lib)
        let compiled_model_ptr = call_dylib_func[CCompiledModel](
            self.engine.lib,
            CompiledModel.CompileModelFnName,
            self.context.borrow_ptr(),
            config.borrow_ptr(),
            status.borrow_ptr(),
        )
        _ = config ^
        if status:
            raise status.__str__()

        return CompiledModel(
            compiled_model_ptr,
            self.engine.lib,
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

        let status = Status(self.engine.lib)
        let model_ptr = call_dylib_func[CModel](
            self.engine.lib,
            Model.InitModelFnName,
            self.context.borrow_ptr(),
            compiled_model.borrow_ptr(),
            status.borrow_ptr(),
        )
        if status:
            raise status.__str__()

        model_ptr.await_model(self.engine.lib)

        let model = Model(
            self.context.borrow_ptr(), model_ptr, self.engine.lib, session ^
        )
        # Keep compiled model alive until model has been awaited.
        _ = compiled_model ^
        return model ^

    fn get_as_engine_tensor_spec(
        self,
        name: StringRef,
        spec: TensorSpec,
        owned session: InferenceSession,
    ) raises -> EngineTensorSpec:
        let context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to create tensor spec"
        return EngineTensorSpec(name, spec, self.engine.lib, session ^)

    fn new_tensor_map(
        self, owned session: InferenceSession
    ) raises -> TensorMap:
        let context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to create tensor map"
        return TensorMap(self.context.borrow_ptr(), self.engine.lib, session ^)


@value
@register_passable
struct InferenceSession:
    var ptr: Pointer[_InferenceSessionImpl]

    fn __init__(
        device: StringRef = "cpu", num_threads: Optional[Int] = None
    ) raises -> Self:
        let path = get_engine_path()
        let self = Self._allocateAndInit(
            path._strref_dangerous(), device, num_threads
        )
        path._strref_keepalive()
        return Self {ptr: self}

    @staticmethod
    fn _allocateAndInit(
        lib_path: StringRef,
        device: StringRef,
        num_threads: Optional[Int],
    ) raises -> Pointer[_InferenceSessionImpl]:
        let ptr = Pointer[_InferenceSessionImpl].alloc(1)
        __get_address_as_uninit_lvalue(ptr.address).__init__(
            lib_path, device, num_threads
        )
        return ptr

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
    ) raises -> EngineTensorSpec:
        return __get_address_as_lvalue(
            self.ptr.address
        ).get_as_engine_tensor_spec(name, spec, self.copy())

    fn new_tensor_map(self) raises -> TensorMap:
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
