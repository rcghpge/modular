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


# CHECK-LABEL: test_fail_list_index_uint
fn test_fail_list_index_uint():
    print("== test_fail_list_index_uint")
    # CHECK: index: 4 is out of bounds for `List` of length: 3
    nums = [1, 2, 3]
    print(nums[UInt(4)])

    # CHECK-NOT: is never reached
    print("is never reached")


fn main():
    test_fail_list_index_uint()
