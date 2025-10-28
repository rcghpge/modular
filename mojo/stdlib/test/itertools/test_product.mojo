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

from itertools import product
from testing import TestSuite, assert_equal, assert_false, assert_true


def test_product2():
    var l1 = ["hey", "hi", "hello"]
    var l2 = [10, 20, 30]

    var it = product(l1, l2)

    var elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 30)

    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 30)

    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 30)

    assert_true(not it.__has_next__())


def test_product2_param():
    var trip_count = 0

    @parameter
    for i, j in product(range(2), range(2)):
        assert_true(i in (0, 1))
        assert_true(j in (0, 1))
        trip_count += 1

    assert_equal(trip_count, 4)


def test_product2_unequal():
    """Checks the product if the two input iterators are unequal."""
    var l1 = ["hey", "hi", "hello", "holla"]
    var l2 = [10, 20, 30]

    var it = product(l1, l2)

    var elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 30)

    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 30)

    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 30)

    elem = next(it)
    assert_equal(elem[0], "holla")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "holla")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "holla")
    assert_equal(elem[1], 30)

    assert_true(not it.__has_next__())


def test_product3():
    """Tests the product of three iterables."""
    var l1 = [1, 2]
    var l2 = [10, 20]
    var l3 = [100, 200]

    var it = product(l1, l2, l3)

    # Expected order: (1,10,100), (1,10,200), (1,20,100), (1,20,200),
    #                 (2,10,100), (2,10,200), (2,20,100), (2,20,200)
    var elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], 10)
    assert_equal(elem[2], 100)

    elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], 10)
    assert_equal(elem[2], 200)

    elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], 20)
    assert_equal(elem[2], 100)

    elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], 20)
    assert_equal(elem[2], 200)

    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], 10)
    assert_equal(elem[2], 100)

    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], 10)
    assert_equal(elem[2], 200)

    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], 20)
    assert_equal(elem[2], 100)

    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], 20)
    assert_equal(elem[2], 200)

    assert_true(not it.__has_next__())


def test_product3_param():
    """Tests the product of three iterables with parameter for loop."""
    var trip_count = 0

    @parameter
    for i, j, k in product(range(2), range(2), range(2)):
        assert_true(i in (0, 1))
        assert_true(j in (0, 1))
        assert_true(k in (0, 1))
        trip_count += 1

    assert_equal(trip_count, 8)


def test_product4():
    """Tests the product of four iterables."""
    var l1 = [1, 2]
    var l2 = [3, 4]
    var l3 = [5, 6]
    var l4 = [7, 8]

    var count = 0
    for a, b, c, d in product(l1, l2, l3, l4):
        assert_true(a in (1, 2))
        assert_true(b in (3, 4))
        assert_true(c in (5, 6))
        assert_true(d in (7, 8))
        count += 1

    # Should have 2 * 2 * 2 * 2 = 16 elements
    assert_equal(count, 16)


def test_product4_param():
    """Tests the product of four iterables with parameter for loop."""
    var trip_count = 0

    @parameter
    for i, j, k, l in product(range(2), range(2), range(2), range(2)):
        assert_true(i in (0, 1))
        assert_true(j in (0, 1))
        assert_true(k in (0, 1))
        assert_true(l in (0, 1))
        trip_count += 1

    assert_equal(trip_count, 16)


def test_product4_order():
    """Tests that product(a, b, c, d) produces elements in the correct order."""
    var l1 = [0, 1]
    var l2 = [0, 1]
    var l3 = [0, 1]
    var l4 = [0, 1]

    var it = product(l1, l2, l3, l4)

    # Rightmost iterator varies fastest
    var elem = next(it)
    assert_equal(elem[0], 0)
    assert_equal(elem[1], 0)
    assert_equal(elem[2], 0)
    assert_equal(elem[3], 0)

    elem = next(it)
    assert_equal(elem[0], 0)
    assert_equal(elem[1], 0)
    assert_equal(elem[2], 0)
    assert_equal(elem[3], 1)

    elem = next(it)
    assert_equal(elem[0], 0)
    assert_equal(elem[1], 0)
    assert_equal(elem[2], 1)
    assert_equal(elem[3], 0)


def test_product_bounds():
    var it = product(range(3), range(2))
    for i in range(6, -1, -1):
        assert_equal(it.bounds()[0], i)
        assert_equal(it.bounds()[1].value(), i)
        var _ = next(it)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
