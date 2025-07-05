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
# RUN: %mojo %s


from testing import assert_equal, assert_raises, assert_true


def test_span_list_int():
    var l = [1, 2, 3, 4, 5, 6, 7]
    var s = Span(list=l)
    assert_equal(len(s), len(l))
    for i in range(len(s)):
        assert_equal(l[i], s[i])
    # subslice
    var s2 = s[2:]
    assert_equal(s2[0], l[2])
    assert_equal(s2[1], l[3])
    assert_equal(s2[2], l[4])
    assert_equal(s2[3], l[5])
    assert_equal(s[-1], l[-1])

    # Test mutation
    s[0] = 9
    assert_equal(s[0], 9)
    assert_equal(l[0], 9)

    s[-1] = 0
    assert_equal(s[-1], 0)
    assert_equal(l[-1], 0)


def test_span_list_str():
    var l = ["a", "b", "c", "d", "e", "f", "g"]
    var s = Span(l)
    assert_equal(len(s), len(l))
    for i in range(len(s)):
        assert_equal(l[i], s[i])
    # subslice
    var s2 = s[2:]
    assert_equal(s2[0], l[2])
    assert_equal(s2[1], l[3])
    assert_equal(s2[2], l[4])
    assert_equal(s2[3], l[5])

    # Test mutation
    s[0] = "h"
    assert_equal(s[0], "h")
    assert_equal(l[0], "h")

    s[-1] = "i"
    assert_equal(s[-1], "i")
    assert_equal(l[-1], "i")


def test_span_array_int():
    var l = InlineArray[Int, 7](1, 2, 3, 4, 5, 6, 7)
    var s = Span[Int](array=l)
    assert_equal(len(s), len(l))
    for i in range(len(s)):
        assert_equal(l[i], s[i])
    # subslice
    var s2 = s[2:]
    assert_equal(s2[0], l[2])
    assert_equal(s2[1], l[3])
    assert_equal(s2[2], l[4])
    assert_equal(s2[3], l[5])

    # Test mutation
    s[0] = 9
    assert_equal(s[0], 9)
    assert_equal(l[0], 9)

    s[-1] = 0
    assert_equal(s[-1], 0)
    assert_equal(l[-1], 0)


def test_span_array_str():
    var l = InlineArray[String, 7]("a", "b", "c", "d", "e", "f", "g")
    var s = Span[String](array=l)
    assert_equal(len(s), len(l))
    for i in range(len(s)):
        assert_equal(l[i], s[i])
    # subslice
    var s2 = s[2:]
    assert_equal(s2[0], l[2])
    assert_equal(s2[1], l[3])
    assert_equal(s2[2], l[4])
    assert_equal(s2[3], l[5])

    # Test mutation
    s[0] = "h"
    assert_equal(s[0], "h")
    assert_equal(l[0], "h")

    s[-1] = "i"
    assert_equal(s[-1], "i")
    assert_equal(l[-1], "i")


def test_indexing():
    var l = InlineArray[Int, 7](1, 2, 3, 4, 5, 6, 7)
    var s = Span[Int](array=l)
    assert_equal(s[True], 2)
    assert_equal(s[Int(0)], 1)
    assert_equal(s[3], 4)


def test_span_slice():
    def compare(s: Span[Int], l: List[Int]) -> Bool:
        if len(s) != len(l):
            return False
        for i in range(len(s)):
            if s[i] != l[i]:
                return False
        return True

    var l = [1, 2, 3, 4, 5]
    var s = Span(l)
    var res = s[1:2]
    assert_equal(res[0], 2)
    res = s[1:-1:1]
    assert_equal(res[0], 2)
    assert_equal(res[1], 3)
    assert_equal(res[2], 4)


def test_copy_from():
    var a = [0, 1, 2, 3]
    var b = [4, 5, 6, 7, 8, 9, 10]
    var s = Span(a)
    var s2 = Span(b)
    s.copy_from(s2[: len(a)])
    for i in range(len(a)):
        assert_equal(a[i], b[i])
        assert_equal(s[i], s2[i])


def test_bool():
    var l = InlineArray[String, 7]("a", "b", "c", "d", "e", "f", "g")
    var s = Span[String](l)
    assert_true(s)
    assert_true(not s[0:0])


def test_contains():
    items = List[Byte](1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    span = Span(items)
    assert_true(0 not in span)
    assert_true(16 not in span)
    for item in items:
        assert_true(item in span)


def test_equality():
    var l = InlineArray[String, 7]("a", "b", "c", "d", "e", "f", "g")
    var l2 = [String("a"), "b", "c", "d", "e", "f", "g"]
    var sp = Span[String](l)
    var sp2 = Span[String](l)
    var sp3 = Span(l2)
    # same pointer
    assert_true(sp == sp2)
    # different pointer
    assert_true(sp == sp3)
    # different length
    assert_true(sp != sp3[:-1])
    # empty
    assert_true(sp[0:0] == sp3[0:0])


def test_fill():
    var a = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    var s = Span(a)

    s.fill(2)

    for i in range(len(a)):
        assert_equal(a[i], 2)
        assert_equal(s[i], 2)


def test_ref():
    var l = InlineArray[Int, 3](1, 2, 3)
    var s = Span[Int](array=l)
    assert_true(s.as_ref() == Pointer(to=l.unsafe_ptr()[]))


def test_reversed():
    var forward = InlineArray[Int, 3](1, 2, 3)
    var backward = InlineArray[Int, 3](3, 2, 1)
    var s = Span[Int](forward)
    var i = 0
    for num in reversed(s):
        assert_equal(num, backward[i])
        i += 1


# We don't actually need to call this test
# but we want to make sure it compiles
def test_span_coerce():
    var l = [1, 2, 3]
    var a = InlineArray[Int, 3](1, 2, 3)

    fn takes_span(s: Span[Int]):
        pass

    takes_span(l)
    takes_span(a)


# We don't actually need to call this test
# but we want to make sure it compiles
def test_conditional_conformance():
    var l = [1, 2, 3]
    var s = Span[Int, alignment=2](l)
    s.fill(0)
    _ = s == s


def test_swap_elements():
    var l = [1, 2, 3, 4, 5]
    var s = Span(l)
    s.swap_elements(1, 4)
    assert_equal(l[1], 5)
    assert_equal(l[4], 2)

    var l2 = ["hi", "hello", "hey"]
    var s2 = Span(l2)
    s2.swap_elements(0, 2)
    assert_equal(l2[0], "hey")
    assert_equal(l2[2], "hi")

    with assert_raises(contains="index out of bounds"):
        s2.swap_elements(0, 4)


def test_merge():
    var a = [1, 2, 3]
    var b = [4, 5, 6]

    fn inner(cond: Bool, mut a: List[Int], mut b: List[Int]):
        var either = Span(a) if cond else Span(b)
        either[0] = 0
        either[-1] = 10

    inner(True, a, b)
    inner(False, a, b)

    assert_equal(a, [0, 2, 10])
    assert_equal(b, [0, 5, 10])


def main():
    test_span_list_int()
    test_span_list_str()
    test_span_array_int()
    test_span_array_str()
    test_indexing()
    test_span_slice()
    test_equality()
    test_bool()
    test_contains()
    test_fill()
    test_ref()
    test_reversed()
    test_swap_elements()
    test_merge()
