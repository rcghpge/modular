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

from math import inf, isinf, isnan

from testing import assert_equal, assert_raises, assert_true


def test_basic_parsing():
    """Test basic parsing functionality."""
    assert_equal(atof("123"), 123.0)
    assert_equal(atof("123.456"), 123.456)
    assert_equal(atof("-123.456"), -123.456)
    assert_equal(atof("+123.456"), 123.456)


def test_scientific_notation():
    """Test scientific notation parsing, which contained the primary bug."""
    assert_equal(atof("1.23e2"), 123.0)
    assert_equal(atof("1.23e+2"), 123.0)
    assert_equal(atof("1.23e-2"), 0.0123)
    assert_equal(atof("1.23E2"), 123.0)
    assert_equal(atof("1.23E+2"), 123.0)
    assert_equal(atof("1.23E-2"), 0.0123)


def test_nan_and_inf():
    """Test NaN and infinity parsing."""
    assert_true(isnan(atof("nan")))
    assert_true(isnan(atof("NaN")))
    assert_true(isinf(atof("inf")))
    assert_true(isinf(atof("infinity")))
    assert_true(isinf(atof("-inf")))
    assert_true(atof("-inf") < 0)
    assert_true(isinf(atof("-infinity")))


def test_leading_decimal():
    """Test parsing with leading decimal point."""
    assert_equal(atof(".123"), 0.123)
    assert_equal(atof("-.123"), -0.123)
    assert_equal(atof("+.123"), 0.123)


def test_trailing_f():
    """Test parsing with trailing 'f'."""
    assert_equal(atof("123.456f"), 123.456)
    assert_equal(atof("123.456F"), 123.456)


def test_large_exponents():
    """Test handling of large exponents."""
    assert_equal(atof("1e309"), inf[DType.float64]())
    assert_equal(atof("1e-309"), 1e-309)


def test_error_cases():
    """Test error cases."""
    with assert_raises(
        contains=(
            "String is not convertible to float: 'abc'. The first character of"
            " 'abc' should be a digit or dot to convert it to a float."
        )
    ):
        _ = atof("abc")

    with assert_raises(contains="String is not convertible to float"):
        _ = atof("")

    with assert_raises(contains="String is not convertible to float"):
        _ = atof(".")

    # TODO:
    # This should actually work and approximate to the closest float64
    # but we don't support it yet. See the section
    # 11, "Processing long numbers quickly" in the paper
    # Number Parsing at a Gigabyte per Second by Daniel Lemire
    # https://arxiv.org/abs/2101.11408 to learn how to do it.
    with assert_raises(
        contains="The number is too long, it's not supported yet."
    ):
        _ = atof("47421763.548648646474532187448684")


alias T = Tuple[Float64, String]
alias numbers_to_test = [
    T(5e-324, "5e-324"),  # smallest value possible with float64
    T(1e-309, "1e-309"),  # subnormal float64
    T(84.5e-309, "84.5e-309"),  # subnormal float64
    T(1e-45, "1e-45"),  # smallest float32 value,
    # largest value possible
    T(1.7976931348623157e308, "1.7976931348623157e+308"),
    T(3.4028235e38, "3.4028235e38"),  # largest value possible, float32
    T(15038927332917.156, "15038927332917.156"),  # triggers step 19
    T(9000000000000000.5, "9000000000000000.5"),  # tie to even
    T(456.7891011e70, "456.7891011e70"),  # Lemire algorithm
    T(0.0, "5e-600"),  # approximate to 0
    T(FloatLiteral.infinity, "5e1000"),  # approximate to infinity
    T(5484.2155e-38, "5484.2155e-38"),  # Lemire algorithm
    T(5e-35, "5e-35"),  # Lemire algorithm
    T(5e30, "5e30"),  # Lemire algorithm
    T(47421763.54884, "47421763.54884"),  # Clinger fast path
    T(474217635486486e10, "474217635486486e10"),  # Clinger fast path
    T(474217635486486e-10, "474217635486486e-10"),  # Clinger fast path
    T(474217635486486e-20, "474217635486486e-20"),  # Clinger fast path
    T(4e-22, "4e-22"),  # Clinger fast path
    T(4.5e15, "4.5e15"),  # Clinger fast path
    T(0.1, "0.1"),  # Clinger fast path
    T(0.2, "0.2"),  # Clinger fast path
    T(0.3, "0.3"),  # Clinger fast path
    # largest uint64 * 10 ** 10
    T(18446744073709551615e10, "18446744073709551615e10"),
    T(3.5e18, "3.5e18"),
    # Examples for issue https://github.com/modularml/mojo/issues/3419
    T(3.5e19, "3.5e19"),
    T(3.5e20, "3.5e20"),
    T(3.5e21, "3.5e21"),
    T(3.5e-15, "3.5e-15"),
    T(3.5e-16, "3.5e-16"),
    T(3.5e-17, "3.5e-17"),
    T(3.5e-18, "3.5e-18"),
    T(3.5e-19, "3.5e-19"),
    T(47421763.54864864647, "47421763.54864864647"),
    # TODO: Make atof work when many digits are present, e.g.
    # "47421763.548648646474532187448684",
]


def test_atof_generate_cases():
    for number, number_as_str in numbers_to_test:
        for suffix in ["", "f", "F"]:
            for exponent in ["e", "E"]:
                for multiplier in ["", "-"]:
                    var sign: Float64 = 1
                    if multiplier == "-":
                        sign = -1
                    final_string = number_as_str.replace("e", exponent)
                    final_string = multiplier + final_string + suffix
                    final_value = sign * number

                    assert_equal(atof(final_string), final_value)


def main():
    test_basic_parsing()
    test_scientific_notation()
    test_nan_and_inf()
    test_leading_decimal()
    test_trailing_f()
    test_large_exponents()
    test_error_cases()
    test_atof_generate_cases()
