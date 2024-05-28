# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s
from tensor import TensorSpec

from max.engine import InferenceSession, InputSpec
from max.serve.service import FileModel
from max.serve.kserve.server import GRPCInferenceServer, MuxInferenceService
from runtime.llcl import (
    Runtime,
    TaskGroup,
    TaskGroupTask,
    TaskGroupTaskList,
    Atomic,
)
from time import sleep, now
from max.serve.kserve.client import GRPCInferenceClient
from max.tensor import Tensor, TensorShape
from max.graph._testing import assert_tensors_almost_equal


def main() -> None:
    var server_started = Atomic[DType.int64](0)
    var server_session = InferenceSession()
    alias server_address = "0.0.0.0:8000"
    var start_time = now()

    @parameter
    fn elapsed() -> String:
        return str(int((now() - start_time) / (1_000_000))) + " ms"

    var server = GRPCInferenceServer.create(server_address, server_session)
    print("[" + elapsed() + "]Created server.")

    @parameter
    async fn server_impl(name: StringRef, path: StringRef) -> None:
        # Load models during service creation:
        try:
            var models = List(FileModel(name, "0", path, None))
            print("[" + elapsed() + "]Entered stub.")
            var mux_service = MuxInferenceService(models, server_session)
            print("[" + elapsed() + "]Loaded model.")
            # Create service and start listening:
            mux_service.init(server)
            print(
                "[" + elapsed() + "]Listening on port " + server_address + "!"
            )
            _ = server_started.fetch_add(1)
            server.serve(mux_service)
        except e:
            print("Server execption: ", e)

    @parameter
    async fn sleeper() -> None:
        sleep(1)

    var model_name = "Add"
    var model_path = "SDK/test/serve/models/add.onnx"
    with Runtime() as rt:
        var tg = TaskGroup(rt)
        var server_task = tg.create_task[NoneType](
            server_impl(model_name, model_path)
        )
        while not server_started.load():
            var sleeper_task = rt.create_task[NoneType](sleeper())
            print(".", end="")
            sleeper_task.wait()
        server.signal_stop()

        tg.wait()
        _ = tg^
        _ = server_task^
        _ = rt
        _ = server^
        print("[" + elapsed() + "]Server terminated succesfully.")
