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
from testing import assert_true


def main():
    # start-basic-example
    alias addOne[x: Int]: Int = x + 1
    alias nine = addOne[8]
    assert_true(nine == 9)
    # end-basic-example

    # start-function-example
    fn add_one(a: Int) -> Int:
        return a + 1

    alias ten = add_one(9)
    # end-function-example
    assert_true(ten == 10)

    # this example includes a an error case, and must be edited in manually.
    # fn return_type() -> AnyType:
    #     return Int  # dynamic type values not permitted yet

    alias IntType = Int

    # start-type-examples
    alias TwoOfAKind[dt: DType] = SIMD[dt, 2]
    twoFloats = TwoOfAKind[DType.float32](1.0, 2.0)

    alias StringKeyDict[ValueType: Copyable & Movable] = Dict[String, ValueType]
    var b: StringKeyDict[UInt8] = {"answer": 42}
    # end-type-examples
    assert_true(twoFloats[0] == 1.0)
    assert_true(twoFloats[1] == 2.0)
    assert_true(b["answer"] == 42)

    # start-floats-example
    alias Floats[size: Int, half_width: Bool = False] = SIMD[
        (DType.float16 if half_width else DType.float32), size
    ]
    var floats = Floats[2](6.0, 8.0)
    var half_floats = Floats[2, True](10.0, 12.0)
    # end-floats-example
    assert_true(floats[0] == 6.0)
    assert_true(floats[1] == 8.0)
    assert_true(half_floats[0] == 10.0)
    assert_true(half_floats[1] == 12.0)
