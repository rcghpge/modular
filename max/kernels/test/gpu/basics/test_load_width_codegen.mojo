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

from gpu.host.compile import _compile_code, get_gpu_target
from gpu.host.info import GPUInfo
from memory.unsafe_pointer import UnsafePointer
from sys.info import simd_width_of
from testing import assert_true, assert_equal

alias _TargetType = __mlir_type.`!kgen.target`


fn kernel(src: UnsafePointer[Float32], dst: UnsafePointer[Float32]):
    var v = src.load[width=8, alignment=32]()
    dst.store[width=8, alignment=32](v)


fn test_kernel_load_32B_width[target: _TargetType]() raises:
    var asm = _compile_code[kernel, target=target]().asm
    assert_true(("v4.b64" in asm) or ("v8.b32" in asm))


fn test_kernel_load_16B_width[target: _TargetType]() raises:
    var asm = _compile_code[kernel, target=target]().asm
    assert_true(("v2.b64" in asm) or ("v4.b32" in asm))


fn main() raises:
    test_kernel_load_16B_width[get_gpu_target["sm_80"]()]()
    test_kernel_load_16B_width[get_gpu_target["sm_90a"]()]()
    test_kernel_load_32B_width[get_gpu_target["sm_100a"]()]()
