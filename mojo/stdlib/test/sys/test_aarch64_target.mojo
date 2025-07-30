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

from sys import simdbitwidth
from sys.info import CompilationTarget

from testing import assert_equal, assert_false, assert_true


def test_arch_query():
    assert_true(CompilationTarget.has_neon())

    assert_equal(simdbitwidth(), 128)

    assert_false(CompilationTarget.has_avx512f())


def main():
    test_arch_query()
