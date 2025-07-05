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
# RUN: %mojo -debug-level full %s

from builtin._location import _SourceLocation
from python import PythonObject
from testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_is,
    assert_is_not,
    assert_not_equal,
    assert_raises,
    assert_true,
)

from utils.numerics import inf, nan


def test_assert_messages():
    assertion = "test_assertion.mojo:"
    assertion_error = ": AssertionError:"
    try:
        assert_true(False)
    except e:
        assert_true(assertion in String(e) and assertion_error in String(e))

    try:
        assert_false(True)
    except e:
        assert_true(assertion in String(e) and assertion_error in String(e))

    try:
        assert_equal(1, 0)
    except e:
        assert_true(assertion in String(e) and assertion_error in String(e))

    try:
        assert_not_equal(0, 0)
    except e:
        assert_true(assertion in String(e) and assertion_error in String(e))


@fieldwise_init
struct DummyStruct(EqualityComparable, Stringable):
    var value: Int

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    @no_inline
    fn __str__(self) -> String:
        return "Dummy"  # Can't be used for equality


def test_assert_equal_is_generic():
    assert_equal(DummyStruct(1), DummyStruct(1))

    with assert_raises():
        assert_equal(DummyStruct(1), DummyStruct(2))


def test_assert_not_equal_is_generic():
    assert_not_equal(DummyStruct(1), DummyStruct(2))

    with assert_raises():
        assert_not_equal(DummyStruct(1), DummyStruct(1))


def test_assert_equal_with_simd():
    assert_equal(SIMD[DType.uint8, 2](1, 1), SIMD[DType.uint8, 2](1, 1))

    with assert_raises():
        assert_equal(SIMD[DType.uint8, 2](1, 1), SIMD[DType.uint8, 2](1, 2))


def test_assert_equal_with_list():
    assert_equal(
        ["This", "is", "Mojo"],
        List("This", "is", "Mojo"),
    )

    with assert_raises():
        assert_equal(
            ["This", "is", "Mojo"],
            List("This", "is", "mojo"),
        )


def test_assert_not_equal_with_list():
    assert_not_equal([3, 2, 1], [3, 1, 0])

    with assert_raises():
        assert_not_equal([3, 2, 1], [3, 2, 1])


def test_assert_almost_equal():
    alias float_type = DType.float32
    alias _inf = inf[float_type]()
    alias _nan = nan[float_type]()

    @parameter
    def _should_succeed[
        dtype: DType, size: Int
    ](
        lhs: SIMD[dtype, size],
        rhs: SIMD[dtype, size],
        *,
        atol: Float64 = 0,
        rtol: Float64 = 0,
        equal_nan: Bool = False,
    ):
        var msg = "`test_assert_almost_equal` should have succeeded"
        assert_almost_equal(
            lhs, rhs, msg=msg, atol=atol, rtol=rtol, equal_nan=equal_nan
        )

    _should_succeed(
        SIMD[float_type, 2](-_inf, _inf), SIMD[float_type, 2](-_inf, _inf)
    )
    _should_succeed(
        SIMD[float_type, 2](-_nan, _nan),
        SIMD[float_type, 2](-_nan, _nan),
        equal_nan=True,
    )
    _should_succeed(
        SIMD[float_type, 2](1.0, -1.1),
        SIMD[float_type, 2](1.1, -1.0),
        atol=0.11,
    )
    _should_succeed(
        SIMD[float_type, 2](1.0, -1.1),
        SIMD[float_type, 2](1.1, -1.0),
        rtol=0.10,
    )

    @parameter
    def _should_fail[
        dtype: DType, size: Int
    ](
        lhs: SIMD[dtype, size],
        rhs: SIMD[dtype, size],
        *,
        atol: Float64 = 0,
        rtol: Float64 = 0,
        equal_nan: Bool = False,
    ):
        var msg = "`test_assert_almost_equal` should have failed"
        with assert_raises(contains=msg):
            assert_almost_equal(
                lhs, rhs, msg=msg, atol=atol, rtol=rtol, equal_nan=equal_nan
            )

    _should_fail(
        SIMD[float_type, 2](-_inf, 0.0),
        SIMD[float_type, 2](_inf, 0.0),
        rtol=0.1,
    )
    _should_fail(
        SIMD[float_type, 2](_inf, 0.0),
        SIMD[float_type, 2](0.0, 0.0),
        rtol=0.1,
    )
    _should_fail(
        SIMD[float_type, 2](_nan, 0.0),
        SIMD[float_type, 2](_nan, 0.0),
        equal_nan=False,
    )
    _should_fail(
        SIMD[float_type, 2](_nan, 0.0),
        SIMD[float_type, 2](0.0, 0.0),
        equal_nan=False,
    )
    _should_fail(
        SIMD[float_type, 2](_nan, 0.0),
        SIMD[float_type, 2](0.0, 0.0),
        equal_nan=True,
    )
    _should_fail(
        SIMD[float_type, 2](1.0, 0.0),
        SIMD[float_type, 2](1.1, 0.0),
        atol=0.05,
    )
    _should_fail(
        SIMD[float_type, 2](-1.0, 0.0),
        SIMD[float_type, 2](-1.1, 0.0),
        rtol=0.05,
    )


def test_assert_is():
    var a = PythonObject("mojo")
    var b = a
    assert_true(a is b)


def test_assert_is_not():
    var a = PythonObject("mojo")
    var b = PythonObject("mojo")
    assert_true(a is not b)


def test_assert_custom_location():
    var location = _SourceLocation(2, 0, "my_file_location.mojo")
    try:
        assert_true(
            False,
            msg="always_false",
            location=location,
        )
    except e:
        assert_true(String(location) in String(e))
        assert_true("always_false" in String(e))


def test_assert_equal_stringslice():
    str1 = StaticString("This is Mojo")
    str2 = "This is Mojo"
    str3 = StaticString("This is mojo")

    fn _build(value: StaticString, start: Int, end: Int) -> StaticString:
        return StaticString(ptr=value.unsafe_ptr() + start, length=end - start)

    fn _build(
        read value: String, start: Int, end: Int
    ) -> StringSlice[__origin_of(value)]:
        return StringSlice[__origin_of(value)](
            ptr=value.unsafe_ptr() + start, length=end - start
        )

    l1 = [_build(str1, 0, 4), _build(str1, 5, 7), _build(str1, 8, 12)]
    l2 = [_build(str2, 0, 4), _build(str2, 5, 7), _build(str2, 8, 12)]
    l3 = [_build(str3, 0, 4), _build(str3, 5, 7), _build(str3, 8, 12)]
    assert_equal(l1, l1)
    assert_equal(l2, l2)
    assert_equal(l1, l2)

    with assert_raises():
        assert_equal(l1, l3)

    with assert_raises():
        assert_equal(l2, l3)


def main():
    test_assert_equal_is_generic()
    test_assert_not_equal_is_generic()
    test_assert_equal_with_simd()
    test_assert_equal_with_list()
    test_assert_not_equal_with_list()
    test_assert_messages()
    test_assert_almost_equal()
    test_assert_is()
    test_assert_is_not()
    test_assert_custom_location()
    test_assert_equal_stringslice()
