# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s
from tensor import TensorSpec

from max.engine import InferenceSession, InputSpec
from max.serve.kserve import FileModel, GRPCServer, MuxInferenceService
from runtime.llcl import (
    Runtime,
    TaskGroup,
    Atomic,
)
from time import sleep, now
from max.graph._testing import assert_tensors_almost_equal


def run_shutdown_test() -> None:
    var server_started = Atomic[DType.int64](0)
    var server_session = InferenceSession()
    alias server_address = "0.0.0.0:0"
    var start_time = now()

    @parameter
    fn elapsed() -> String:
        return str(int((now() - start_time) / (1_000_000))) + " ms"

    var server = GRPCServer.create(server_address, server_session)
    print("[" + elapsed() + "]Created server.")

    @parameter
    async fn server_impl():
        # N.B. This is simple startup and shutdown test. We don't actually need
        # to load any models here, so we just provide the empty list.
        try:
            var models = List[FileModel]()
            print("[" + elapsed() + "]Entered stub.")
            var mux_service = MuxInferenceService(models, server_session)
            print("[" + elapsed() + "]Loaded model.")
            print(
                "[" + elapsed() + "]Listening on port " + server_address + "!"
            )
            _ = server_started.fetch_add(1)
            server.serve(mux_service)
        except e:
            print("Server execption: ", e)

    @parameter
    async fn shutdown_task_impl():
        while not server_started.load():
            print(".", end="")
            sleep(1)
        server.signal_stop()

    with Runtime() as rt:
        var tg = TaskGroup[__lifetime_of(server_started)](rt)
        tg.create_task(server_impl())
        tg.create_task(shutdown_task_impl())
        tg.wait()
        _ = tg^
        _ = server^
        print("[" + elapsed() + "]Server terminated succesfully.")


def main():
    run_shutdown_test()
