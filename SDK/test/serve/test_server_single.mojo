# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -debug-level full %s

import sys

from tensor import Tensor, TensorShape

from max.engine import InferenceSession
from max.graph import _testing
from max.serve.kserve.client import GRPCInferenceClient


fn test_basic_add_request(
    inout client: GRPCInferenceClient, owned session: InferenceSession
) raises:
    var inputs = session.new_tensor_map()
    var a = Tensor[DType.float32](1, 1)
    var b = Tensor[DType.float32](1, 1)
    a[0][0] = 1
    b[0][0] = 1
    inputs.borrow("args_0", a)
    inputs.borrow("args_1", b)

    var req_outputs = List[String]("output_0")
    var response = client.infer("Add", "0", inputs, req_outputs)
    _ = a^
    _ = b^

    var actual = response.get_output_tensors().get[DType.float32](
        req_outputs[0]
    )
    var expected = Tensor[DType.float32](TensorShape(1, 1), 2)  # 1 + 1 = 2
    _testing.assert_tensors_almost_equal(
        actual,
        expected,
        atol=1e-4,
        rtol=1e-4,
    )
    _ = response^


fn main() raises:
    var USAGE = "Usage: test_single.mojo [-a address] [-n iterations]"
    var argv = sys.argv()

    var i = 1
    var address: String = "localhost:8000"
    var num_requests = 1
    while i < len(argv):
        if argv[i] == "-a":
            address = str(argv[i + 1]).strip()
            if not address:
                raise Error("Must specify non-empty address")
            i += 2
        elif argv[i] == "-n":
            num_requests = atol(argv[i + 1])
            i += 2
        else:
            print(USAGE)
            raise Error("Failed to parse command-line options")

    var session = InferenceSession()
    var client = GRPCInferenceClient(address, session)

    for i in range(num_requests):
        try:
            test_basic_add_request(client, session)
        except e:
            print(e)
