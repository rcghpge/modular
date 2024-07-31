# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo %s
# REQUIRES: manual

from max.tensor import TensorSpec, Tensor

from max.engine import InferenceSession, InputSpec
from max.serve.kserve import FileModel, MuxInferenceService
from time import sleep, now
from max.serve.kserve import GRPCClient
from max.graph._testing import assert_tensors_almost_equal
from max.serve.kserve.types import InferenceRequest, InferenceResponse


def main():
    var server_address = "localhost:0"  # shouldn't need an address
    var session = InferenceSession()
    var client = GRPCClient(server_address, session)
    var name = "BasicAdd"
    var path = "SDK/test/API/serve/models/add.onnx"

    var inputs = session.new_tensor_map()
    var a = Tensor[DType.float32](1, 1)
    var b = Tensor[DType.float32](1, 1)
    var res = Tensor[DType.float32](1, 1)
    a[0][0] = 1
    b[0][0] = 2
    res[0][0] = a[0][0] + b[0][0]
    inputs.borrow("args_0", a)
    inputs.borrow("args_1", b)

    var outputs = List[String]("output_0")
    var request = client._make_inference_request(name, "0", inputs, outputs)
    var models = List(FileModel(name, "0", path, None))
    var mux_service = MuxInferenceService(models, session)
    var resp = mux_service.infer(request)
    assert_tensors_almost_equal(res, resp.get[DType.float32]("output_0"))
    _ = client^
    _ = inputs^
    _ = a^
    _ = b^
