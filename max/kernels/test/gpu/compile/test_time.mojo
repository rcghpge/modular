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

from time import perf_counter_ns, time_function

from gpu.host.compile import _compile_code
from gpu.host import get_gpu_target
from gpu.intrinsics import *
from testing import *


fn clock_functions():
    _ = perf_counter_ns()


@always_inline
fn _verify_clock_functions(asm: StringSlice) raises -> None:
    assert_false("mov.u32" in asm)
    assert_true("mov.u64" in asm)


def test_clock_functions_sm80():
    var asm = _compile_code[
        clock_functions, target = get_gpu_target["sm_80"]()
    ]().asm
    _verify_clock_functions(asm)


def test_clock_functions_sm90():
    var asm = _compile_code[
        clock_functions, target = get_gpu_target["sm_90"]()
    ]().asm
    _verify_clock_functions(asm)


fn time_functions(some_value: Int) -> Int:
    var tmp = some_value

    @always_inline
    @parameter
    fn something():
        tmp += 1

    _ = time_function[something]()

    return tmp


@always_inline
fn _verify_time_functions(asm: StringSlice) raises -> None:
    assert_true("mov.u64" in asm)
    assert_true("add.s64" in asm)


def test_time_functions_sm80():
    var asm = _compile_code[
        time_functions, target = get_gpu_target["sm_80"]()
    ]().asm
    _verify_time_functions(asm)


def test_time_functions_sm90():
    var asm = _compile_code[
        time_functions, target = get_gpu_target["sm_90"]()
    ]().asm
    _verify_time_functions(asm)


def main():
    test_clock_functions_sm80()
    test_clock_functions_sm90()
    test_time_functions_sm80()
    test_time_functions_sm90()
