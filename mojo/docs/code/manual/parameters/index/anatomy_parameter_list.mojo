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


# Docs example *only* includes the signature, to show the elements of a
# parameter list.
fn my_sort[
    # infer-only parameters
    Type: DType,
    width: Int, //,
    # positional-only parameter
    values: SIMD[Type, width],
    /,
    # positional-or-keyword parameter
    compare: fn (Scalar[Type], Scalar[Type]) -> Int,
    *,
    # keyword-only parameter
    reverse: Bool = False,
]() -> SIMD[Type, width]:
    sorted = SIMD[Type, width](values)
    for output_position in range(width):
        lowest = output_position
        for compare_position in range(output_position + 1, width):
            if compare(sorted[lowest], sorted[compare_position]) == 1:
                lowest = compare_position
        if lowest != output_position:
            sorted[output_position], sorted[lowest] = (
                sorted[lowest],
                sorted[output_position],
            )
    return sorted


def main():
    alias dtype = DType.int32
    alias input2 = SIMD[dtype, 8](9, 3, 3, 1, 11, 10, 5, 2)

    fn compare(lhs: Scalar[dtype], rhs: Scalar[dtype]) -> Int:
        if lhs == rhs:
            return 0
        else:
            return -1 if lhs < rhs else 1

    alias sorted = my_sort[input2, compare, reverse=False]()
    print(sorted)
