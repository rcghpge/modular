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

from math import sqrt


fn rsqrt[dt: DType, width: Int](x: SIMD[dt, width]) -> SIMD[dt, width]:
    return 1 / sqrt(x)


struct One[Type: Writable & Copyable & Movable]:
    var value: Type

    fn __init__(out self, value: Type):
        self.value = value


def use_one():
    s1 = One(123)
    s2 = One("Hello")


struct Two[Type: Writable & Copyable & Movable]:
    var val1: Type
    var val2: Type

    fn __init__(out self, var one: One[Type], var another: One[Type]):
        self.val1 = one.value
        self.val2 = another.value
        print(String(self.val1), String(self.val2))

    @staticmethod
    fn fire(thing1: One[Type], thing2: One[Type]):
        print("🔥", String(thing1.value), String(thing2.value))


def use_two():
    s3 = Two(One("infer"), One("me"))
    Two.fire(One(1), One(2))
    # Two.fire(One("mixed"), One(0)) # Error: parameter inferred to two different values


def main():
    var v = SIMD[DType.float16, 4](33)
    print(rsqrt(v))

    # second example
    use_one()

    # third example
    use_two()
