# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import sys

from memory import memcpy
from python import Python, PythonObject

from max.tensor import Tensor, TensorShape
from max.engine.tensor import EngineNumpyView
from max.graph import _testing


fn to_tensor[
    dtype: DType
](inout np_array: PythonObject) raises -> Tensor[dtype]:
    var view = EngineNumpyView(np_array)
    var size = view.spec().num_elements()
    var ptr = UnsafePointer[Scalar[dtype]].alloc(size)
    memcpy(ptr, view.unsafe_ptr().bitcast[dtype](), size)
    return Tensor[dtype](view.spec(), ptr)


fn test_basic_add_request(
    np: PythonObject, triton_grpc: PythonObject, client: PythonObject
) raises:
    var inputs: PythonObject = []
    var outputs: PythonObject = []
    _ = inputs.append(triton_grpc.InferInput("args_0", [1, 1], "FP32"))
    _ = inputs.append(triton_grpc.InferInput("args_1", [1, 1], "FP32"))
    _ = outputs.append(triton_grpc.InferRequestedOutput("output_0"))

    var arr: PythonObject = []
    _ = arr.append([1.0])
    _ = inputs[0].set_data_from_numpy(np.array(arr, np.float32))
    _ = inputs[1].set_data_from_numpy(np.array(arr, np.float32))

    var results = client.infer("Add", inputs, "0", outputs)
    var output0_arr = results.as_numpy("output_0")
    var actual = to_tensor[DType.float32](output0_arr)
    var expected = Tensor[DType.float32](TensorShape(1, 1), 2)  # 1 + 1 = 2
    _testing.assert_tensors_almost_equal(
        actual,
        expected,
        atol=1e-4,
        rtol=1e-4,
    )
    print("verified")


fn main() raises:
    var argv = sys.argv()
    var USAGE = str("Usage: mojo ") + argv[0] + " [-a address] [-n iterations]"

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

    var np = Python.import_module("numpy")
    var triton_grpc = Python.import_module("tritonclient.grpc")
    var client = triton_grpc.InferenceServerClient(address)

    for _ in range(num_requests):
        try:
            test_basic_add_request(np, triton_grpc, client)
        except e:
            print(e)
