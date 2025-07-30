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
#
# This file only tests that conversion of negative IntLiteral to UInt fails.
#
# ===----------------------------------------------------------------------=== #
# RUN: not %bare-mojo %s 2>&1 | FileCheck %s


fn main():
    # CHECK: test_uint_error.mojo:[[@LINE+1]]:{{.*}}: error: integer value -1 is negative, but is being converted to an unsigned type
    print(UInt(-1))
    # CHECK-NOT: is never reached
    print("is never reached")
