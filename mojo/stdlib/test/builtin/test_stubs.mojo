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

from sys.intrinsics import _type_is_eq

from testing import assert_false, assert_true, TestSuite


def test_uint_for():
    for _i in range(UInt(5)):
        assert_false(_type_is_eq[type_of(_i), Int]())
        assert_true(_type_is_eq[type_of(_i), UInt]())

    for _i in range(UInt(1), UInt(5)):
        assert_false(_type_is_eq[type_of(_i), Int]())
        assert_true(_type_is_eq[type_of(_i), UInt]())

    for _i in range(UInt(1), UInt(5), UInt(2)):
        assert_false(_type_is_eq[type_of(_i), Int]())
        assert_true(_type_is_eq[type_of(_i), UInt]())


def test_uint_parameter_for():
    @parameter
    for _i in range(UInt(5)):
        assert_false(_type_is_eq[type_of(_i), Int]())
        assert_true(_type_is_eq[type_of(_i), UInt]())

    @parameter
    for _i in range(UInt(1), UInt(5)):
        assert_false(_type_is_eq[type_of(_i), Int]())
        assert_true(_type_is_eq[type_of(_i), UInt]())

    @parameter
    for _i in range(UInt(1), UInt(5), UInt(2)):
        assert_false(_type_is_eq[type_of(_i), Int]())
        assert_true(_type_is_eq[type_of(_i), UInt]())


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
