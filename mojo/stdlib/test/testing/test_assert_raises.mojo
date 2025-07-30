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

from testing import assert_equal, assert_raises, assert_true


fn test_assert_raises_catches_error() raises:
    with assert_raises():
        raise Error("SomeError")
    # The assert_raises should catch the error and not propagate it.
    # Hence the test will succeed.


fn test_assert_raises_catches_matched_error() raises:
    with assert_raises(contains="Some"):
        raise Error("SomeError")

    with assert_raises(contains="Error"):
        raise Error("SomeError")

    with assert_raises(contains="eE"):
        raise Error("SomeError")


fn test_assert_raises_no_error() raises:
    try:
        with assert_raises():  # col 27
            pass
        raise Error("This should not be reachable.")
    except e:
        assert_true(String(e).startswith("AssertionError: Didn't raise"))
        assert_true(String(e).endswith(":27"))  # col 27
        assert_true(String(e) != "This should not be reachable.")


fn test_assert_raises_no_match() raises:
    try:
        with assert_raises(contains="Some"):
            raise Error("OtherError")
        raise Error("This should not be reachable.")
    except e:
        assert_equal(String(e), "OtherError")


def main():
    test_assert_raises_catches_error()
    test_assert_raises_catches_matched_error()
    test_assert_raises_no_error()
    test_assert_raises_no_match()
