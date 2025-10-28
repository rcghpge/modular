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

from itertools import repeat
from testing import TestSuite, assert_equal, assert_false, assert_true


def test_repeat_finite():
    """Tests repeat with a finite number of times."""
    var it = repeat(42, times=3)

    assert_true(it.__has_next__())
    assert_equal(next(it), 42)
    assert_true(it.__has_next__())
    assert_equal(next(it), 42)
    assert_true(it.__has_next__())
    assert_equal(next(it), 42)
    assert_false(it.__has_next__())


def test_repeat_string():
    """Tests repeat with a string value."""
    var it = repeat("hello", times=5)

    var count = 0
    for val in it:
        assert_equal(val, "hello")
        count += 1

    assert_equal(count, 5)


def test_repeat_zero_times():
    """Tests repeat with zero repetitions."""
    var it = repeat(99, times=0)
    assert_false(it.__has_next__())


def test_repeat_one_time():
    """Tests repeat with a single repetition."""
    var it = repeat(7, times=1)

    assert_true(it.__has_next__())
    assert_equal(next(it), 7)
    assert_false(it.__has_next__())


def test_repeat_large_count():
    """Tests repeat with a large number of repetitions."""
    var it = repeat(123, times=1000)

    var count = 0
    for val in it:
        assert_equal(val, 123)
        count += 1

    assert_equal(count, 1000)


def test_repeat_in_for_loop():
    """Tests repeat iterator in a for loop."""
    var sum = 0
    for val in repeat(10, times=5):
        sum += val

    assert_equal(sum, 50)


def test_repeat_param():
    """Tests repeat with parameter for loop."""
    var trip_count = 0

    @parameter
    for val in repeat(42, times=3):
        assert_equal(val, 42)
        trip_count += 1

    assert_equal(trip_count, 3)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
