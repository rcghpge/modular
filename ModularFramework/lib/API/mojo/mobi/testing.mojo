# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys.param_env import env_get_string
from tensor import Tensor, TensorShape, TensorSpec
from testing import assert_true, assert_almost_equal, assert_equal

from aiengine import EngineTensorView, InferenceSession, TensorMap


alias TEMP_DIR = Path(env_get_string["TEMP_DIR"]())


fn assert_no_dynamic_dims(spec: TensorSpec) raises:
    for i in range(spec.rank()):
        assert_true(spec[i] >= 0)


fn assert_tensors_almost_equal[
    dtype: DType
](
    a: Tensor[dtype],
    b: Tensor[dtype],
    absolute_tolerance: SIMD[dtype, 1] = 1e-8,
    relative_tolerance: SIMD[dtype, 1] = 1e-5,
) raises:
    assert_true(a.spec() == b.spec())
    assert_no_dynamic_dims(a.spec())
    assert_no_dynamic_dims(b.spec())

    for i in range(a.num_elements()):
        assert_almost_equal[dtype](
            a[i], b[i], absolute_tolerance, relative_tolerance
        )


fn assert_tensors_equal[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises:
    assert_true(a.spec() == b.spec())
    assert_no_dynamic_dims(a.spec())
    assert_no_dynamic_dims(b.spec())

    for i in range(a.num_elements()):
        assert_equal[dtype](a[i], b[i])


fn execute_unary[
    intype: DType = DType.float32, outtype: DType = DType.float32
](module: Module, name: StringRef, input: Tensor[intype]) raises -> Tensor[
    outtype
]:
    let result_map = execute(module, name, input)
    return result_map.get[outtype]("output0")


fn execute_binary[
    dtype: DType = DType.float32
](
    module: Module, name: StringRef, x: Tensor[dtype], y: Tensor[dtype]
) raises -> Tensor[dtype]:
    let result_map = execute(module, name, x, y)
    return result_map.get[dtype]("output0")


fn execute(m: Module, name: StringRef, *tensors: Tensor) raises -> TensorMap:
    # TODO: Don't ask the user to provide a unique name, autogenerate it.
    let temp_path = TEMP_DIR / name

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
