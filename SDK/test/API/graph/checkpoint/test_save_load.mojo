# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -D TEMP_FILE_DIR=%T %s

from pathlib import Path
from sys.param_env import env_get_string

from max.graph import _testing
from max.graph.checkpoint import save, load, TensorDict
from tensor import Tensor, TensorShape

alias TEMP_FILE_DIR = env_get_string["TEMP_FILE_DIR"]()


fn test_simple() raises:
    var tensors = TensorDict()
    tensors.set("x", Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4))
    tensors.set("y", Tensor[DType.float32](TensorShape(10, 5), -1.23))

    var TEMP_FILE = Path(TEMP_FILE_DIR) / "test_simple"
    save(tensors, TEMP_FILE)

    var loaded = load(TEMP_FILE)
    _testing.assert_equal(2, len(loaded))
    var x = loaded.get[DType.int32]("x")
    _testing.assert_tensors_equal(
        x,
        Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4),
    )

    var y = loaded.get[DType.float32]("y")
    _testing.assert_tensors_equal(
        y,
        Tensor[DType.float32](TensorShape(10, 5), -1.23),
    )


fn test_weird_keys() raises:
    var tensors = TensorDict()
    tensors.set("", Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4))
    tensors.set(
        "key that's very long",
        Tensor[DType.float32](TensorShape(10, 5), -1.23),
    )
    tensors.set(
        "another/key.with.some.symbols!",
        Tensor[DType.uint64](TensorShape(2), 123, 4567),
    )

    var TEMP_FILE = Path(TEMP_FILE_DIR) / "test_weird_keys"
    save(tensors, TEMP_FILE)

    var loaded = load(TEMP_FILE)
    _testing.assert_equal(3, len(loaded))
    _testing.assert_tensors_equal(
        loaded.get[DType.int32](""),
        Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4),
    )

    _testing.assert_tensors_equal(
        loaded.get[DType.float32]("key that's very long"),
        Tensor[DType.float32](TensorShape(10, 5), -1.23),
    )

    _testing.assert_tensors_equal(
        loaded.get[DType.uint64]("another/key.with.some.symbols!"),
        Tensor[DType.uint64](TensorShape(2), 123, 4567),
    )


fn test_large_tensor() raises:
    var tensors = TensorDict()
    tensors.set("x", Tensor[DType.int32](TensorShape(100, 200, 200), 567))

    var TEMP_FILE = Path(TEMP_FILE_DIR) / "test_large_tensor"
    save(tensors, TEMP_FILE)

    var loaded = load(TEMP_FILE)
    _testing.assert_equal(1, len(loaded))
    var t = loaded.get[DType.int32]("x")
    _testing.assert_tensors_equal(
        t,
        Tensor[DType.int32](TensorShape(100, 200, 200), 567),
    )


def main():
    test_simple()
    test_weird_keys()
    test_large_tensor()
