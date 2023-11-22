# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from aiengine import EngineTensorView, InferenceSession, TensorMap


fn execute(m: Module, temp_path: Path, *tensors: Tensor) raises -> TensorMap:
    with open(temp_path, "w") as f:
        f.write(m.__str__())

    let session = InferenceSession()
    let model = session.load_model(temp_path.path._strref_dangerous())

    let input_map = session.new_tensor_map()
    for i in range(len(tensors)):
        let input_name = "input" + String(i)
        input_map.borrow(
            input_name._strref_dangerous(),
            __get_address_as_lvalue(tensors[i]),
        )
        input_name._strref_keepalive()

    let result_map = model.execute(input_map)
    temp_path.path._strref_keepalive()

    return result_map ^


fn execute(m: Module, temp_path: Path, tensors: TensorMap) raises -> TensorMap:
    with open(temp_path, "w") as f:
        f.write(m.__str__())

    let session = InferenceSession()
    let model = session.load_model(temp_path.path._strref_dangerous())

    let result_map = model.execute(tensors)
    temp_path.path._strref_keepalive()

    return result_map ^
