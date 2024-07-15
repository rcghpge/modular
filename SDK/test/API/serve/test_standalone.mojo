# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo %s
# REQUIRES: manual
from max.tensor import TensorSpec, Tensor, TensorShape

from max.engine import InferenceSession, InputSpec, TensorMap, NamedTensor

from max.serve.kserve.server import GRPCServer
from max.serve.kserve import MuxInferenceService, FileModel
from runtime.llcl import TaskGroup, Atomic
from time import sleep, perf_counter_ns
from max.serve.kserve import GRPCClient
from max.graph._testing import assert_tensors_almost_equal
from testing import assert_true


trait ModelTestCase:
    @staticmethod
    fn get_model_name() -> String:
        ...

    @staticmethod
    fn get_model_config() -> FileModel:
        ...

    @staticmethod
    fn get_model_input(
        session: InferenceSession,
    ) raises -> Tuple[TensorMap, List[NamedTensor]]:
        ...

    @staticmethod
    fn get_output_names() raises -> List[String]:
        ...

    @staticmethod
    fn verify(inout actual: TensorMap) raises:
        ...


###############################################################################
struct RobertaTorch(ModelTestCase):
    @staticmethod
    fn get_model_name() -> String:
        return "roberta"

    @staticmethod
    fn get_model_config() -> FileModel:
        var model_path = "SDK/test/API/serve/models/roberta.torchscript"
        return FileModel(
            Self.get_model_name(),
            "0",
            model_path,
            List(
                InputSpec(TensorSpec(DType.int64, 1, 128)),
                InputSpec(TensorSpec(DType.int64, 1, 128)),
            ),
        )

    @staticmethod
    fn get_model_input(
        session: InferenceSession,
    ) raises -> Tuple[TensorMap, List[NamedTensor]]:
        var inputs = session.new_tensor_map()
        var a = Tensor[DType.float32](1, 128)
        var b = Tensor[DType.float32](1, 128)
        # initialize a,b
        inputs.borrow("input_ids", a)
        inputs.borrow("attention_mask", b)
        var aT = NamedTensor("input_ids", a)
        var bT = NamedTensor("attention_mask", b)
        var t = List[NamedTensor](aT, bT)
        return Tuple(inputs^, t)

    @staticmethod
    fn get_output_names() raises -> List[String]:
        return List[String]("result0")

    @staticmethod
    fn verify(inout actual: TensorMap) raises:
        # TODO implement an interesting roberta test
        pass


###############################################################################
struct BasicAddTorch(ModelTestCase):
    @staticmethod
    fn get_model_name() -> String:
        return "BasicAdd_int32_pt"

    @staticmethod
    fn get_model_config() -> FileModel:
        var model_path = "SDK/test/API/serve/models/add.pt"
        return FileModel(
            Self.get_model_name(),
            "0",
            model_path,
            List(
                InputSpec(TensorSpec(DType.int32, 1, 1)),
                InputSpec(TensorSpec(DType.int32, 1, 1)),
            ),
        )

    @staticmethod
    fn get_model_input(
        session: InferenceSession,
    ) raises -> Tuple[TensorMap, List[NamedTensor]]:
        var inputs = session.new_tensor_map()
        var a = Tensor[DType.int32](1, 1)
        var b = Tensor[DType.int32](1, 1)
        a[0][0] = 1
        b[0][0] = 2
        inputs.borrow("input_1", a)
        inputs.borrow("input_2", b)
        var aT = NamedTensor("input_1", a)
        var bT = NamedTensor("input_2", b)
        var t = List[NamedTensor](aT, bT)
        return Tuple(inputs^, t)

    @staticmethod
    fn get_output_names() raises -> List[String]:
        return List[String]("result0")

    @staticmethod
    fn verify(inout actual: TensorMap) raises:
        var a = Tensor[DType.int32](1, 1)
        a[0][0] = 3
        # FIXME SERV-161 -  uncomment after resolved.
        # assert_tensors_almost_equal(a, actual.get[DType.int32]("result0"))
        pass


###############################################################################


struct BasicAddOnnx(ModelTestCase):
    @staticmethod
    fn get_model_name() -> String:
        return "BasicAdd_int32_onnx"

    @staticmethod
    fn get_model_config() -> FileModel:
        var model_path = "SDK/test/API/serve/models/add.onnx"
        return FileModel(
            Self.get_model_name(),
            "0",
            model_path,
            List(
                InputSpec(TensorSpec(DType.int32, 1, 1)),
                InputSpec(TensorSpec(DType.int32, 1, 1)),
            ),
        )

    @staticmethod
    fn get_model_input(
        session: InferenceSession,
    ) raises -> Tuple[TensorMap, List[NamedTensor]]:
        var inputs = session.new_tensor_map()
        var a = Tensor[DType.int32](1, 1)
        var b = Tensor[DType.int32](1, 1)
        a[0][0] = 1
        b[0][0] = 2
        inputs.borrow("args_0", a)
        inputs.borrow("args_1", b)
        var aT = NamedTensor("args_0", a)
        var bT = NamedTensor("args_1", b)
        var t = List[NamedTensor](aT, bT)
        return Tuple(inputs^, t)

    @staticmethod
    fn get_output_names() raises -> List[String]:
        return List[String]("output_0")

    @staticmethod
    fn verify(inout actual: TensorMap) raises:
        var a = Tensor[DType.int32](1, 1)
        a[0][0] = 3
        # FIXME SERV-161 -  uncomment after resolved.
        # assert_tensors_almost_equal(a, actual.get[DType.int32]("output_0"))
        pass


###############################################################################


def test_standalone_server_client[TEST_CASE: ModelTestCase]() -> None:
    var start_time = perf_counter_ns()

    @parameter
    fn elapsed() -> String:
        return str(int((perf_counter_ns() - start_time) / (1_000_000))) + " ms"

    var server_started = Atomic[DType.int64](0)
    var client_completed = Atomic[DType.int64](0)
    var client_success = False
    var server_success = False
    var server_session = InferenceSession()
    alias server_address = "0.0.0.0:8004"
    var server = GRPCServer.create(server_address, server_session)

    print("[" + elapsed() + "]Created server.")

    @parameter
    async fn server_impl():
        # Load models during service creation:
        try:
            var models = List(TEST_CASE.get_model_config())
            print("[" + elapsed() + "]Loading models")
            var server_service = MuxInferenceService(models, server_session)
            print(
                "[" + elapsed() + "]Listening on port " + server_address + "!"
            )

            @parameter
            fn server_started_fn() capturing -> NoneType:
                print("[" + elapsed() + "]Server started...")
                _ = server_started.fetch_add(1)
                return None

            server.serve[MuxInferenceService, server_started_fn](server_service)
            server_success = True
            _ = models^

        except e:
            print("[" + elapsed() + "]Server error : ", e)

    @parameter
    async fn client_impl():
        try:
            while not server_started.load():
                sleep(1.0)
            # Load models during service creation:
            var session = InferenceSession()
            var client = GRPCClient(server_address, session)
            var input_tensors = TEST_CASE.get_model_input(session)
            var inputs = input_tensors[0]

            var output_names = TEST_CASE.get_output_names()
            print("[" + elapsed() + "]Created inputs" + str(inputs))

            var actual = client.infer(
                TEST_CASE.get_model_name(), "0", inputs, output_names
            )

            print("[" + elapsed() + "]Client end...")
            var result = actual.get_output_tensors()
            TEST_CASE.verify(result)
            _ = actual^
            _ = inputs^
            _ = client_completed.fetch_add(1)
            client_success = True
        except e:
            print("[" + elapsed() + "]Client error : ", e)
            _ = client_completed.fetch_add(1)

    var tg = TaskGroup[__lifetime_of(server)]()
    tg.create_task(server_impl())

    print("[" + elapsed() + "]Created server")

    tg.create_task(client_impl())
    print("[" + elapsed() + "]Created client")
    while not client_completed.load():
        sleep(1)
    server.signal_stop()

    tg.wait()
    print("[" + elapsed() + "]Signalled stop")
    print("[" + elapsed() + "]Done!")
    _ = server^
    _ = server_started^
    _ = server_session^
    assert_true(client_success == True and server_success == True)
    print("=============================================")


fn print_m[TEST_CASE: ModelTestCase]():
    print(TEST_CASE.get_model_name())


def main():
    # test_standalone_server_client[RobertaTorch]()
    test_standalone_server_client[BasicAddOnnx]()
    # test_standalone_server_client[BasicAddTorch]()
    alias m = BasicAddOnnx
    print_m[m]()
    # print(m.get_output_names())
