# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Checks that each GPU vendor's MAX plugin lowers the warp shuffles."""

from std.compile import compile_info
from std.sys.info import CompilationTarget
from std.testing import TestSuite, assert_equal, assert_true

from max.gpu.host import get_gpu_target
from max.gpu.primitives.warp import (
    shuffle_down,
    shuffle_idx,
    shuffle_up,
    shuffle_xor,
)


def _compile_shuffle_kernel[target: CompilationTarget]() raises -> String:
    # Nested so `__functions_in_module()` doesn't compile it for the host,
    # where the shuffles are rejected.
    def kernel(data: Pointer[Float32, MutAnyOrigin]):
        var val = data[]
        data[unsafe_offset=0] = shuffle_idx(val, 1)
        data[unsafe_offset=1] = shuffle_up(val, 1)
        data[unsafe_offset=2] = shuffle_down(val, 1)
        data[unsafe_offset=3] = shuffle_xor(val, 1)

    return String(compile_info[kernel, emission_kind="llvm", target=target]())


def _assert_nvidia_shuffle(
    llvm: String, mnemonic: StaticString, width_mask: Int
) raises:
    var call = String(
        "call float @llvm.nvvm.shfl.sync.", mnemonic, ".f32(i32 -1, float "
    )
    var args_tail = String(", i32 1, i32 ", width_mask, ")")
    for line in llvm.splitlines():
        if call in line:
            assert_true(line.endswith(args_tail), String(line))
            return
    raise Error("no `shfl.sync.", mnemonic, "` call in:\n", llvm)


def _assert_nvidia_shuffles(llvm: String) raises:
    _assert_nvidia_shuffle(llvm, "idx", 31)
    # `shfl.sync.up` takes a zero width mask where the others take 31.
    _assert_nvidia_shuffle(llvm, "up", 0)
    _assert_nvidia_shuffle(llvm, "down", 31)
    _assert_nvidia_shuffle(llvm, "bfly", 31)


def test_nvidia() raises:
    _assert_nvidia_shuffles(
        _compile_shuffle_kernel[get_gpu_target["sm_80"]()]()
    )


def test_nvidia_bare_triple() raises:
    # Named only by its triple, as `--target-triple` builds it: no
    # `stdlib_plugin` field and no plugin payload.
    comptime target = CompilationTarget[
        _mlir_value=__mlir_attr[
            `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
            `arch = "sm_80", `,
            `features = "+ptx81", `,
            `data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
            `simd_bit_width = 128,`,
            `index_bit_width = 64`,
            `> : !kgen.target`,
        ]
    ]()
    _assert_nvidia_shuffles(_compile_shuffle_kernel[target]())


def test_amd() raises:
    var llvm = _compile_shuffle_kernel[get_gpu_target["mi300x"]()]()
    assert_equal(llvm.count("call i32 @llvm.amdgcn.ds.bpermute("), 4)


def test_apple() raises:
    var llvm = _compile_shuffle_kernel[get_gpu_target["apple-m4"]()]()
    assert_true("call float @air.simd_shuffle.f32.i16(float" in llvm)
    assert_true("call float @air.simd_shuffle_up.f32.i16(float" in llvm)
    assert_true("call float @air.simd_shuffle_down.f32.i16(float" in llvm)
    assert_true("call float @air.simd_shuffle_xor.f32.i16(float" in llvm)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
