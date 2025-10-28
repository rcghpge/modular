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

from builtin.simd import *
from gpu.host import DeviceContext
from memory import bitcast


fn print_bits[dtype: DType](val: Scalar[dtype]):
    var u8 = bitcast[DType.uint8](val)
    var bits = String(capacity=32)

    @parameter
    for i in reversed(range(8)):
        bits.write((u8 >> i) & 1)

    print(dtype, "nan:", u8, bits)


fn test():
    # CHECK: float8_e5m2 nan: 127 01111111
    print_bits(Float8_e5m2(FloatLiteral.nan))
    # CHECK: float8_e4m3fn nan: 127 01111111
    print_bits(Float8_e4m3fn(FloatLiteral.nan))


def main():
    with DeviceContext() as ctx:
        ctx.enqueue_function_experimental[test](grid_dim=1, block_dim=1)
