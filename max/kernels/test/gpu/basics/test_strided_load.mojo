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

from sys.intrinsics import strided_load

from gpu.host.compile import _compile_code
from gpu.memory import AddressSpace
from testing import assert_true


fn strided_load_kernel[
    *, dtype: DType = DType.uint32, width: Int = 1
](
    output: UnsafePointer[SIMD[dtype, width]],
    ptr: UnsafePointer[Scalar[dtype], address_space = AddressSpace.GENERIC],
    stride: Int,
):
    output[] = strided_load[width](ptr, stride)


def test_strided_load():
    assert_true(
        "@llvm.masked.gather"
        in _compile_code[strided_load_kernel[width=4], emission_kind="llvm"]()
    )


def main():
    test_strided_load()
