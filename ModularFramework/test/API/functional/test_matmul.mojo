# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo -I %mobi_pkg_dir -I %test_utils_pkg_dir -D TEMP_FILE_DIR=%T "%s"

from sys.param_env import env_get_string
from tensor import Tensor, TensorShape, TensorSpec

from testing import assert_true, assert_almost_equal

from mobi import Module, mobicc
from mobi import functional as G
from mobi import testing


alias TEMP_FILE_DIR = Path(env_get_string["TEMP_FILE_DIR"]())


fn execute[
    dtype: DType = DType.float32
](module: Module, name: StringRef, input: Tensor[dtype]) raises -> Tensor[
    dtype
]:
    let result_map = testing.execute(module, TEMP_FILE_DIR / name, input)
    return result_map.get[dtype]("output0")


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


fn test_matmul() raises:
    alias name = "test_matmul"
    alias dim = 2
    alias hidden_dim = 2

    var m = Module()
    let batch_size = m.dyn()
    let seq_len = m.dyn()
    var g = m.graph(
        name,
        m.types(m.f32(batch_size, seq_len, dim)),
        m.types(m.f32(batch_size, seq_len, hidden_dim)),
    )

    let w = g.constant(
        Tensor[DType.float32](
            TensorShape(hidden_dim, dim),
            0.5641,
            0.4875,
            -1.1172,
            -1.1583,
        )
    )

    # g.output(G.matmul(w, g[0]))
    g.output(G.matmul(g[0], w))

    assert_true(mobicc.module_verify(m.m))

    let input = Tensor[DType.float32](
        TensorShape(2, 2, dim),
        -1.2620,
        -2.0678,
        -1.6634,
        1.3036,
        -0.0088,
        -1.1315,
        1.1287,
        1.7699,
    )

    let expected = Tensor[DType.float32](
        TensorShape(2, 2, hidden_dim),
        1.5983,
        1.7799,
        -2.3947,
        -2.3209,
        1.2591,
        1.3063,
        -1.3406,
        -1.4998,
    )

    let out = execute(m, name, input)

    assert_tensors_almost_equal(expected, out, 1e-4)


fn test_batch_matmul() raises:
    alias name = "test_batch_matmul"
    alias dim = 2
    alias hidden_dim = 2

    var m = Module()
    let batch_size = m.dyn()
    let seq_len = m.dyn()
    var g = m.graph(
        name,
        m.types(m.f32(batch_size, dim)),
        m.types(m.f32(batch_size, m.dyn(), hidden_dim)),
    )

    let w = g.constant(
        Tensor[DType.float32](
            TensorShape(2, 2, dim),
            -1.2620,
            -2.0678,
            -1.6634,
            1.3036,
            -0.0088,
            -1.1315,
            1.1287,
            1.7699,
        )
    )

    # g.output(G.matmul(w, g[0]))
    g.output(G.matmul(g[0], w))

    let input = Tensor[DType.float32](
        TensorShape(hidden_dim, dim),
        0.5641,
        0.4875,
        -1.1172,
        -1.1583,
    )

    assert_true(mobicc.module_verify(m.m))

    let expected = Tensor[DType.float32](
        TensorShape(2, 2, hidden_dim),
        -1.5228,
        -0.5309,
        3.3366,
        0.8002,
        0.5453,
        0.2245,
        -1.2975,
        -0.7860,
    )

    let out = execute(m, name, input)

    assert_tensors_almost_equal(expected, out, 1e-4)


def main():
    test_matmul()
    test_batch_matmul()
