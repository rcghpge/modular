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

import sys


fn main() raises:
    test_write_to_stdout()


@fieldwise_init
struct Point(Writable):
    var x: Int
    var y: Int

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Point(", self.x, ", ", self.y, ")")


# CHECK-LABEL: test_write_to_stdout
fn test_write_to_stdout():
    print("== test_write_to_stdout")

    var stdout = sys.stdout

    # CHECK: Hello, World!
    stdout.write("Hello, World!")

    # CHECK: point = Point(1, 1)
    var point = Point(1, 1)
    stdout.write("point = ", point)
