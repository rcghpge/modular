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

from std.math import cos, exp2, rsqrt, sin, sqrt
from std.testing import assert_almost_equal

from max.gpu import thread_idx
from max.gpu.host import DeviceContext

comptime _N = 8
comptime _UnaryFn = def[dtype: DType, width: SIMDLength, //](
    SIMD[dtype, width]
) thin -> SIMD[dtype, width] where dtype.is_floating_point()


def _input[dtype: DType](i: Int) -> Scalar[dtype]:
    return Scalar[dtype](i + 1) * 0.5


def _kernel[
    func: _UnaryFn, dtype: DType
](
    output: MutPointer[Scalar[dtype], MutAnyOrigin]
) where dtype.is_floating_point():
    output[thread_idx.x] = func(_input[dtype](thread_idx.x))


def _check[
    func: _UnaryFn, name: StaticString, dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    var out_device = ctx.enqueue_create_buffer[dtype](_N)
    ctx.enqueue_function[_kernel[func, dtype]](
        out_device, grid_dim=1, block_dim=_N
    )
    with out_device.map_to_host() as out_host:
        for i in range(_N):
            assert_almost_equal(
                out_host[i].cast[.float32](),
                func(_input[dtype](i)).cast[.float32](),
                msg=String(name, "(", _input[dtype](i), ") for ", dtype),
                rtol=1e-5 if dtype == .float32 else 1e-2,
            )


def _check_all[func: _UnaryFn, name: StaticString](ctx: DeviceContext) raises:
    _check[func, name, .float32](ctx)
    _check[func, name, .float16](ctx)
    _check[func, name, .bfloat16](ctx)


def main() raises:
    with DeviceContext() as ctx:
        _check_all[sqrt, "sqrt"](ctx)
        _check_all[rsqrt, "rsqrt"](ctx)
        _check_all[exp2, "exp2"](ctx)
        _check_all[sin, "sin"](ctx)
        _check_all[cos, "cos"](ctx)
