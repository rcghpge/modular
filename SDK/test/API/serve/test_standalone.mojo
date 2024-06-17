# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo %s
# REQUIRES: manual

from tensor import TensorSpec

from max.engine import InferenceSession, InputSpec
from max.serve.kserve import FileModel, GRPCServer, MuxInferenceService
from runtime.llcl import (
    Runtime,
    TaskGroup,
    Atomic,
)
from time import sleep, now
from max.serve.kserve import GRPCClient
from max.graph._testing import assert_tensors_almost_equal


def test_standalone_server_client() -> None:
    var start_time = now()

    @parameter
    fn elapsed() -> String:
        return str(int((now() - start_time) / (1_000_000))) + " ms"

    var server_started = Atomic[DType.int64](0)
    var client_completed = Atomic[DType.int64](0)
    var server_session = InferenceSession()
    alias server_address = "0.0.0.0:8000"
    var server = GRPCServer.create(server_address, server_session)
    print("[" + elapsed() + "]Created server.")

    @parameter
    async fn server_impl(name: StringRef, path: StringRef) -> None:
        # Load models during service creation:
        try:
            var models = List(FileModel(name, "0", path, None))
            var server_service = MuxInferenceService(models, server_session)
            print(
                "[" + elapsed() + "]Listening on port " + server_address + "!"
            )
            _ = server_started.fetch_add(1)
            server.serve(server_service)
        except e:
            print("Server error : ", e)

    @parameter
    async fn client(name: StringRef) -> None:
        try:
            while not server_started.load():
                sleep(1)
            # Load models during service creation:
            var session = InferenceSession()
            var client = GRPCClient(server_address, session)
            var inputs = session.new_tensor_map()
            var a = Tensor[DType.float32](1, 1)
            var b = Tensor[DType.float32](1, 1)
            a[0][0] = 1
            b[0][0] = 1
            inputs.borrow("args_0", a)
            inputs.borrow("args_1", b)

            var req_outputs = List[String]("output_0")
            print("[" + elapsed() + "]Created inputs")

            var response = client.infer(name, "0", inputs, req_outputs)
            _ = a^
            _ = b^

            var actual = response.get_output_tensors().get[DType.float32](
                req_outputs[0]
            )
            var expected = Tensor[DType.float32](
                TensorShape(1, 1), 2
            )  # 1 + 1 = 2
            assert_tensors_almost_equal(expected, actual)
            print("[" + elapsed() + "]Client end...")
            _ = response^
            _ = expected^
            # signal to stop main loop
            _ = client_completed.fetch_add(1)

        except e:
            print("Client error : ", e)

    var model_name = "Add"
    var model_path = "SDK/test/API/serve/models/add.onnx"
    with Runtime() as rt:
        var tg = TaskGroup[__lifetime_of(model_name, model_path)](rt)
        var server_task = tg.create_task(server_impl(model_name, model_path))
        var client_task = tg.create_task(client(model_name))
        while not client_completed.load():
            sleep(1)
        server.signal_stop()
        tg.wait()
        _ = tg^
        print("[" + elapsed() + "]Done!")
        _ = server^
        _ = rt


def main():
    test_standalone_server_client()
