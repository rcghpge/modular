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

from sys.info import simd_bit_width, simd_width_of

from gpu.host import get_gpu_target
from testing import assert_equal


def test_simd_bit_width():
    assert_equal(128, simd_bit_width[target = get_gpu_target()]())
    assert_equal(4, simd_width_of[Float32, target = get_gpu_target()]())


def main():
    test_simd_bit_width()
