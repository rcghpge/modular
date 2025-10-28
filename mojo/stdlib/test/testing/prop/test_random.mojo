# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from os import abort
from testing import (
    assert_raises,
    assert_equal,
    assert_true,
    assert_false,
    TestSuite,
)
from testing.prop.random import Rng


def test_rng_xoshiro_float():
    var rng = Rng(seed=1234)
    for _ in range(100):
        var f = rng._xoshiro_float()
        assert_true(f >= 0.0)
        assert_true(f <= 1.0)


def test_rng_rand_bool():
    var rng = Rng(seed=1234)
    for _ in range(100):
        assert_true(rng.rand_bool(true_probability=1.0))
        assert_false(rng.rand_bool(true_probability=0.0))


def test_rng_rand_scalar():
    def test_dtype[dtype: DType](min: Scalar[dtype], max: Scalar[dtype]):
        var rng = Rng(seed=1234)
        for _ in range(100):
            var value = rng.rand_scalar[dtype](min=min, max=max)
            assert_true(value >= min)
            assert_true(value <= max)

    @parameter
    for dtype in [
        DType.uint,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
        DType.int,
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.float16,
        DType.float32,
        DType.float64,
    ]:
        alias scalar = Scalar[dtype]

        test_dtype[dtype](scalar.MIN_FINITE, scalar.MAX_FINITE)

        @parameter
        if dtype.is_signed():
            test_dtype[dtype](scalar(-10), scalar(10))
        else:
            test_dtype[dtype](scalar(10), scalar(20))


def test_rng_rand_scalar_raises():
    with assert_raises(contains="invalid min/max"):
        var rng = Rng(seed=1234)
        var _ = rng.rand_scalar[DType.int32](min=10, max=5)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
