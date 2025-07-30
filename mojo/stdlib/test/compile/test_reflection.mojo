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

from compile.reflection import get_linkage_name, get_type_name
from testing import assert_equal
from sys.info import _current_target


fn my_func() -> Int:
    return 0


def test_get_linkage_name():
    var name = get_linkage_name[my_func]()
    assert_equal(name, "test_reflection::my_func()")


def test_get_linkage_name_nested():
    fn nested_func(x: Int) -> Int:
        return x

    var name2 = get_linkage_name[nested_func]()
    assert_equal(
        name2,
        "test_reflection::test_get_linkage_name_nested()_nested_func(::Int)",
    )


fn your_func[x: Int]() raises -> Int:
    return x


def test_get_linkage_name_parameterized():
    var name = get_linkage_name[your_func[7]]()
    assert_equal(name, "test_reflection::your_func[::Int](),x=7")


def test_get_linkage_name_on_itself():
    var name = get_linkage_name[_current_target]()
    assert_equal(name, "stdlib::sys::info::_current_target()")


def test_get_type_name():
    var name = get_type_name[Int]()
    assert_equal(name, "stdlib.builtin.int.Int")


def test_get_type_name_nested():
    fn nested_func[T: AnyType]() -> StaticString:
        return get_type_name[T]()

    var name = nested_func[String]()
    assert_equal(name, "stdlib.collections.string.string.String")


def main():
    test_get_linkage_name()
    test_get_linkage_name_nested()
    test_get_linkage_name_parameterized()
    test_get_linkage_name_on_itself()
    test_get_type_name()
    test_get_type_name_nested()
