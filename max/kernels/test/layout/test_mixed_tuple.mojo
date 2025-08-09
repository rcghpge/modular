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
"""Tests for the unified LayoutLike system."""

from testing import assert_equal
from layout._mixed_tuple import MixedIntTuple, Idx, ComptimeInt, RuntimeInt
from sys import sizeof


fn test_nested_layouts() raises:
    print("== test_nested_layouts")

    # Create nested layouts
    var inner = MixedIntTuple(Idx[2](), Idx(3))
    var nested = MixedIntTuple(inner, Idx[4]())
    assert_equal(inner[1].value(), 3)
    assert_equal(nested[0][0].value(), 2)
    assert_equal(nested[1].value(), 4)
    assert_equal(sizeof[__type_of(inner)](), sizeof[Int]())
    assert_equal(sizeof[__type_of(nested)](), sizeof[Int]())


fn test_list_literal_construction() raises:
    print("== test_list_literal_construction")
    var t: MixedIntTuple[ComptimeInt[2], RuntimeInt[DType.index]] = [
        Idx[2](),
        Idx(3),
    ]
    assert_equal(t[0].value(), 2)
    assert_equal(t[1].value(), 3)


fn main() raises:
    test_nested_layouts()
    test_list_literal_construction()
