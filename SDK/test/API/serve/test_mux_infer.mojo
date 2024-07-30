# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo %s %add_onnx

from max.tensor import TensorSpec, Tensor

from max.engine import InferenceSession, InputSpec
from max.serve.kserve import FileModel, MuxInferenceService
from time import sleep, now
from max.serve.kserve import GRPCClient
from max.graph._testing import assert_tensors_almost_equal
from max.serve.kserve.types import InferenceRequest, InferenceResponse
from pathlib import cwd, Path
from testing import assert_false
from os import env
from sys import argv
from utils.index import Index


# alias ModelPath = "../API/serve/models/add.onnx"
alias ModelName = "BasicAdd"


fn test_mux_init(path: String) raises:
    """Tests the mux inference server initialization."""
    var session = InferenceSession()
    var name = ModelName
    var models = List(FileModel(name, "0", path, None))
    var mux_service = MuxInferenceService(models, session)
    _ = mux_service^


fn test_mux_infer(path: String) raises:
    """Test a single request via mux-inference server without going through the network.
    """
    var server_address = "localhost:0"  # shouldn't need an address
    var session = InferenceSession()
    var client = GRPCClient(server_address, session)
    var name = ModelName
    var inputs = session.new_tensor_map()
    var a = Tensor[DType.int32](1, 1)
    var b = Tensor[DType.int32](1, 1)
    var res = Tensor[DType.int32](1, 1)
    a[Index(0, 0)] = 1
    b[Index(0, 0)] = 2
    res[Index(0, 0)] = a[Index(0, 0)] + b[Index(0, 0)]

    # for add.pt
    # inputs.borrow("input_1", a)
    # inputs.borrow("input_2", b)
    # var output_name = "result_0"

    # for add.onnx
    inputs.borrow("args_0", a)
    inputs.borrow("args_1", b)
    var output_name = "output_0"

    var outputs = List[String](output_name)
    var request = client._make_inference_request(name, "0", inputs, outputs)
    var models = List(FileModel(name, "0", path, None))
    var mux_service = MuxInferenceService(models, session)
    var resp = mux_service.infer(request)
    assert_tensors_almost_equal(
        res, resp.get[DType.int32](output_name), atol=0, rtol=0
    )
    _ = client^
    _ = inputs^
    _ = a^
    _ = b^


def main():
    if len(argv()) < 2:
        assert_false("needs path to add.onnx")
    else:
        # CMAKE hack - cleanup when cmake is no longer needed
        if argv()[1] == "%add_onnx":
            var model_path = "../API/serve/models/add.onnx"
            test_mux_init(model_path)
            test_mux_infer(model_path)
        else:
            test_mux_init(argv()[1])
            test_mux_infer(argv()[1])
