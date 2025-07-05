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

from sys.ffi import c_char

from testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
)


def test_add():
    assert_equal("five", StringLiteral.__add__("five", ""))
    assert_equal("six", StringLiteral.__add__("", "six"))
    assert_equal("fivesix", StringLiteral.__add__("five", "six"))


def test_mul():
    alias `3`: Int = 3
    alias `u3`: UInt = 3
    alias static_concat_0 = "mojo" * 3
    alias static_concat_1 = "mojo" * `3`
    alias static_concat_2 = "mojo" * `u3`
    assert_equal(static_concat_0, static_concat_1)
    assert_equal(static_concat_1, static_concat_2)
    assert_equal("mojomojomojo", static_concat_0)
    assert_equal(static_concat_0, "mojo" * 3)
    var dynamic_concat = "mojo" * 3
    assert_equal("mojomojomojo", dynamic_concat)
    assert_equal(static_concat_0, dynamic_concat)


def test_equality():
    assert_false(StringLiteral.__eq__("five", "six"))
    assert_true(StringLiteral.__eq__("six", "six"))

    assert_true(StringLiteral.__ne__("five", "six"))
    assert_false(StringLiteral.__ne__("six", "six"))

    var hello = "hello"
    var hello_ref = hello.as_string_slice()

    assert_false(StringLiteral.__eq__("goodbye", hello))
    assert_true(StringLiteral.__eq__("hello", hello))

    assert_false(StringLiteral.__eq__("goodbye", hello_ref))
    assert_true(StringLiteral.__eq__("hello", hello_ref))


def test_len():
    assert_equal(0, StringLiteral.__len__(""))
    assert_equal(4, StringLiteral.__len__("four"))


def test_bool():
    assert_true(StringLiteral.__bool__("not_empty"))
    assert_false(StringLiteral.__bool__(""))


def test_find():
    assert_equal(0, "Hello world".find(""))
    assert_equal(0, "Hello world".find("Hello"))
    assert_equal(2, "Hello world".find("llo"))
    assert_equal(6, "Hello world".find("world"))
    assert_equal(-1, "Hello world".find("universe"))

    assert_equal(3, "...a".find("a", 0))
    assert_equal(3, "...a".find("a", 1))
    assert_equal(3, "...a".find("a", 2))
    assert_equal(3, "...a".find("a", 3))

    # Test find() support for negative start positions
    assert_equal(4, "Hello world".find("o", -10))
    assert_equal(7, "Hello world".find("o", -5))

    assert_equal(-1, "abc".find("abcd"))


def test_rfind():
    # Basic usage.
    assert_equal("hello world".rfind("world"), 6)
    assert_equal("hello world".rfind("bye"), -1)

    # Repeated substrings.
    assert_equal("ababab".rfind("ab"), 4)

    # Empty string and substring.
    assert_equal("".rfind("ab"), -1)
    assert_equal("foo".rfind(""), 3)

    # Test that rfind(start) returned pos is absolute, not relative to specified
    # start. Also tests positive and negative start offsets.
    assert_equal("hello world".rfind("l", 5), 9)
    assert_equal("hello world".rfind("l", -5), 9)
    assert_equal("hello world".rfind("w", -3), -1)
    assert_equal("hello world".rfind("w", -5), 6)

    assert_equal(-1, "abc".rfind("abcd"))


def test_startswith():
    var str = "Hello world"

    assert_true(str.startswith("Hello"))
    assert_false(str.startswith("Bye"))

    assert_true(str.startswith("llo", 2))
    assert_true(str.startswith("llo", 2, -1))
    assert_false(str.startswith("llo", 2, 3))


def test_endswith():
    var str = "Hello world"

    assert_true(str.endswith(""))
    assert_true(str.endswith("world"))
    assert_true(str.endswith("ld"))
    assert_false(str.endswith("universe"))

    assert_true(str.endswith("ld", 2))
    assert_true(str.endswith("llo", 2, 5))
    assert_false(str.endswith("llo", 2, 3))


def test_comparison_operators():
    # Test less than and greater than
    assert_true(StringLiteral.__lt__("abc", "def"))
    assert_false(StringLiteral.__lt__("def", "abc"))
    assert_false(StringLiteral.__lt__("abc", "abc"))
    assert_true(StringLiteral.__lt__("ab", "abc"))
    assert_true(StringLiteral.__gt__("abc", "ab"))
    assert_false(StringLiteral.__gt__("abc", "abcd"))

    # Test less than or equal to and greater than or equal to
    assert_true(StringLiteral.__le__("abc", "def"))
    assert_true(StringLiteral.__le__("abc", "abc"))
    assert_false(StringLiteral.__le__("def", "abc"))
    assert_true(StringLiteral.__ge__("abc", "abc"))
    assert_false(StringLiteral.__ge__("ab", "abc"))
    assert_true(StringLiteral.__ge__("abcd", "abc"))

    # Test case sensitivity in comparison (assuming ASCII order)
    assert_true(StringLiteral.__gt__("abc", "ABC"))
    assert_false(StringLiteral.__le__("abc", "ABC"))

    # Test comparisons involving empty strings
    assert_true(StringLiteral.__lt__("", "abc"))
    assert_false(StringLiteral.__lt__("abc", ""))
    assert_true(StringLiteral.__le__("", ""))
    assert_true(StringLiteral.__ge__("", ""))

    # Test less than and greater than
    def_slice = "def".as_string_slice()
    abcd_slice = "abc".as_string_slice()
    assert_true(StringLiteral.__lt__("abc", def_slice))
    assert_false(StringLiteral.__lt__("def", abcd_slice[0:3]))
    assert_false(StringLiteral.__lt__("abc", abcd_slice[0:3]))
    assert_true(StringLiteral.__lt__("ab", abcd_slice[0:3]))
    assert_true(StringLiteral.__gt__("abc", abcd_slice[0:2]))
    assert_false(StringLiteral.__gt__("abc", abcd_slice))

    # Test less than or equal to and greater than or equal to
    assert_true(StringLiteral.__le__("abc", def_slice))
    assert_true(StringLiteral.__le__("abc", abcd_slice[0:3]))
    assert_false(StringLiteral.__le__("def", abcd_slice[0:3]))
    assert_true(StringLiteral.__ge__("abc", abcd_slice[0:3]))
    assert_false(StringLiteral.__ge__("ab", abcd_slice[0:3]))
    assert_true(StringLiteral.__ge__("abcd", abcd_slice[0:3]))

    abc_upper_slice = "ABC".as_string_slice()
    # Test case sensitivity in comparison (assuming ASCII order)
    assert_true(StringLiteral.__gt__("abc", abc_upper_slice))
    assert_false(StringLiteral.__le__("abc", abc_upper_slice))

    empty_slice = "".as_string_slice()
    # Test comparisons involving empty strings
    assert_true(StringLiteral.__lt__("", abcd_slice[0:3]))
    assert_false(StringLiteral.__lt__("abc", empty_slice))
    assert_true(StringLiteral.__le__("", empty_slice))
    assert_true(StringLiteral.__ge__("", empty_slice))


def test_indexing():
    var s = "hello"
    assert_equal(s[False], "h")
    assert_equal(s[Int(1)], "e")
    assert_equal(s[2], "l")


def test_intable():
    assert_equal(StringLiteral.__int__("123"), 123)

    with assert_raises():
        _ = StringLiteral.__int__("hi")


def test_isdigit():
    assert_true("123".isdigit())
    assert_false("abc".isdigit())
    assert_false("123abc".isdigit())
    # TODO: Uncomment this when PR3439 is merged
    # assert_false("".isdigit())


def test_islower():
    assert_true("hello".islower())
    assert_false("Hello".islower())
    assert_false("HELLO".islower())
    assert_false("123".islower())
    assert_false("".islower())


def test_isupper():
    assert_true("HELLO".isupper())
    assert_false("Hello".isupper())
    assert_false("hello".isupper())
    assert_false("123".isupper())
    assert_false("".isupper())


def test_iter():
    # Test iterating over a string
    var s = "one"
    var i = 0
    for c in s.codepoints():
        if i == 0:
            assert_equal(String(c), "o")
        elif i == 1:
            assert_equal(String(c), "n")
        elif i == 2:
            assert_equal(String(c), "e")


def test_layout():
    # Test empty StringLiteral contents
    var empty = "".unsafe_ptr()
    # An empty string literal is stored as just the NUL terminator.
    assert_true(Int(empty) != 0)
    # TODO(MSTDL-596): This seems to hang?
    # assert_equal(empty[0], 0)

    # Test non-empty StringLiteral C string
    var ptr: UnsafePointer[c_char] = "hello".unsafe_cstr_ptr()
    assert_equal(ptr[0], ord("h"))
    assert_equal(ptr[1], ord("e"))
    assert_equal(ptr[2], ord("l"))
    assert_equal(ptr[3], ord("l"))
    assert_equal(ptr[4], ord("o"))
    assert_equal(ptr[5], 0)  # Verify NUL terminated


def test_lower_upper():
    assert_equal("hello".lower(), "hello")
    assert_equal("HELLO".lower(), "hello")
    assert_equal("Hello".lower(), "hello")
    assert_equal("hello".upper(), "HELLO")
    assert_equal("HELLO".upper(), "HELLO")
    assert_equal("Hello".upper(), "HELLO")


def test_repr():
    # Usual cases
    assert_equal(StringLiteral.__repr__("hello"), "'hello'")

    # Escape cases
    assert_equal(StringLiteral.__repr__("\0"), r"'\x00'")
    assert_equal(StringLiteral.__repr__("\x06"), r"'\x06'")
    assert_equal(StringLiteral.__repr__("\x09"), r"'\t'")
    assert_equal(StringLiteral.__repr__("\n"), r"'\n'")
    assert_equal(StringLiteral.__repr__("\x0d"), r"'\r'")
    assert_equal(StringLiteral.__repr__("\x0e"), r"'\x0e'")
    assert_equal(StringLiteral.__repr__("\x1f"), r"'\x1f'")
    assert_equal(StringLiteral.__repr__(" "), "' '")
    assert_equal(StringLiteral.__repr__("'"), '"\'"')
    assert_equal(StringLiteral.__repr__("A"), "'A'")
    assert_equal(StringLiteral.__repr__("\\"), r"'\\'")
    assert_equal(StringLiteral.__repr__("~"), "'~'")
    assert_equal(StringLiteral.__repr__("\x7f"), r"'\x7f'")


def test_strip():
    assert_equal("".strip(), "")
    assert_equal("  ".strip(), "")
    assert_equal("  hello".strip(), "hello")
    assert_equal("hello  ".strip(), "hello")
    assert_equal("  hello  ".strip(), "hello")
    assert_equal("  hello  world  ".strip(" "), "hello  world")
    assert_equal("_wrap_hello world_wrap_".strip("_wrap_"), "hello world")
    assert_equal("  hello  world  ".strip("  "), "hello  world")
    assert_equal("  hello  world  ".lstrip(), "hello  world  ")
    assert_equal("  hello  world  ".rstrip(), "  hello  world")
    assert_equal(
        "_wrap_hello world_wrap_".lstrip("_wrap_"), "hello world_wrap_"
    )
    assert_equal(
        "_wrap_hello world_wrap_".rstrip("_wrap_"), "_wrap_hello world"
    )


def test_count():
    var str = "Hello world"

    assert_equal(12, str.count(""))
    assert_equal(1, str.count("Hell"))
    assert_equal(3, str.count("l"))
    assert_equal(1, str.count("ll"))
    assert_equal(1, str.count("ld"))
    assert_equal(0, str.count("universe"))

    assert_equal(String("aaaaa").count("a"), 5)
    assert_equal(String("aaaaaa").count("aa"), 3)


def test_rjust():
    assert_equal("hello".rjust(4), "hello")
    assert_equal("hello".rjust(8), "   hello")
    assert_equal("hello".rjust(8, "*"), "***hello")


def test_ljust():
    assert_equal("hello".ljust(4), "hello")
    assert_equal("hello".ljust(8), "hello   ")
    assert_equal("hello".ljust(8, "*"), "hello***")


def test_center():
    assert_equal("hello".center(4), "hello")
    assert_equal("hello".center(8), " hello  ")
    assert_equal("hello".center(8, "*"), "*hello**")


def test_float_conversion():
    assert_equal(("4.5").__float__(), 4.5)
    assert_equal(Float64("4.5"), 4.5)
    with assert_raises():
        _ = ("not a float").__float__()


def main():
    test_add()
    test_mul()
    test_equality()
    test_len()
    test_bool()
    test_find()
    test_rfind()
    test_comparison_operators()
    test_count()
    test_indexing()
    test_intable()
    test_isdigit()
    test_islower()
    test_isupper()
    test_layout()
    test_lower_upper()
    test_repr()
    test_rjust()
    test_ljust()
    test_center()
    test_startswith()
    test_endswith()
    test_strip()
    test_float_conversion()
