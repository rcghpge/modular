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
"""Test for MixedTuple GPU memory codegen."""

import sys

from gpu.host.compile import _compile_code, get_gpu_target
from layout._mixed_tuple import ComptimeInt, Idx, MixedTuple, RuntimeInt
from memory.unsafe_pointer import UnsafePointer
from testing import assert_true


fn kernel(v: Int, ptr: UnsafePointer[Int32]):
    """Kernel that uses MixedTuple with both compile-time and runtime indices.

    Args:
        v: Runtime value for dynamic indexing.
        ptr: Output pointer to store results.
    """
    var l = MixedTuple(Idx[1](), MixedTuple(Idx(v), Idx[3]()))
    ptr[0] = Int32(l[0].value())
    ptr[1] = Int32(l[1][0].value())


fn test_mixed_tuple_codegen_memory() raises:
    var amd_asm = _compile_code[
        kernel, target = get_gpu_target["amdgpu:gfx942"]()
    ]().asm

    assert_true("buffer_load_dword" not in amd_asm)
    assert_true("v_mov_b32_e32 v0, 1" in amd_asm)
    assert_true("v_mov_b32_e32 v1, s0" in amd_asm)

    var nvidia_asm = _compile_code[
        kernel, target = get_gpu_target["sm_80"]()
    ]().asm

    assert_true("ld.local" not in nvidia_asm)
    assert_true("st.global.b32 \t[%rd3], 1" in nvidia_asm)
    assert_true("st.global.b32 \t[%rd3+4], %rd1" in nvidia_asm)


fn main() raises:
    test_mixed_tuple_codegen_memory()
