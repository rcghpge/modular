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

# RUN: not %mojo %s 2>&1 | FileCheck %s

from memory import UnsafePointerV2


fn test_cannot_cast_between_different_named_origins[
    T: AnyType, mut: Bool, //, origin: Origin[mut]
](p: UnsafePointerV2[T, origin]):
    pass


def main():
    var x = 42
    var y = 55

    var p = UnsafePointerV2(to=x)
    # CHECK: argument #0 cannot be converted from 'UnsafePointerV2[Int, x]' to 'UnsafePointerV2[Int, y]'
    test_cannot_cast_between_different_named_origins[origin_of(y)](p)
