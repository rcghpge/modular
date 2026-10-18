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

from std._plugin._trait import (
    PluginHooks,
    StdPlugin,
    _PrintEmitPluginHookFnType,
    _UnaryMathPluginHookFnType,
)
from std.collections import OptionalReg
from std.sys.info import CompilationTarget
from std.collections.string.string_span import _get_kgen_string
from std.ffi import CStringSpan
from std.io.file_descriptor import RawFileDescriptor
from std.io.io import _printf
from std.reflection.location import SourceLocation

from .cuda_math import _has_nvptx_op, _nvptx_unary_math


def _print_emit[O: Origin](cstr: CStringSpan[O], file: RawFileDescriptor):
    """Writes `print()` output from an NVIDIA kernel through device `printf`.

    Device `printf` has a single output stream, so `file` is ignored.
    """
    _printf["%s"](cstr.ptr())


@inline(.always)
def _debug_assert_emit[
    O: Origin
](message: Pointer[Byte, O], length: Int, loc: SourceLocation):
    """Writes a failed `debug_assert` message, with the failing block and
    thread, through device `printf`.
    """
    # `_debug_assert_msg` hands every assertion to this hook, including ones
    # evaluated at compile time, so it keeps the stdlib's compile-time message.
    if __is_run_in_comptime_interpreter:
        print("At: ", loc, ": Assert Error: ", message, sep="")
        return

    from std._gpu.primitives.id import block_idx, thread_idx

    _printf[
        "At: %s:%llu:%llu: block: [%llu,%llu,%llu] thread:"
        " [%llu,%llu,%llu] Assert Error: %s\n"
    ](
        loc.file_name().as_c_string_span(),
        loc.line(),
        loc.column(),
        UInt(block_idx.x),
        UInt(block_idx.y),
        UInt(block_idx.z),
        UInt(thread_idx.x),
        UInt(thread_idx.y),
        UInt(thread_idx.z),
        message,
    )


struct CUDAPlugin(StdPlugin):
    """`PluginHooks` implementation for NVIDIA CUDA backends."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["cuda"]()
    comptime HookFor[target: CompilationTarget] = CUDAPluginHooks[target]


struct CUDAPluginHooks[target: CompilationTarget](PluginHooks):
    comptime print_emit_fn: OptionalReg[
        _PrintEmitPluginHookFnType
    ] = _print_emit

    @staticmethod
    @inline(.always)
    def debug_assert_emit_fn[
        O: Origin
    ](message: Pointer[Byte, O], length: Int, loc: SourceLocation):
        _debug_assert_emit(message, length, loc)

    comptime _handles_debug_assert: Bool = True

    comptime unary_math_fn[
        name: StaticString, dtype: DType, width: Int
    ]: OptionalReg[_UnaryMathPluginHookFnType] = (
        _nvptx_unary_math[name=name] if _has_nvptx_op[
            name, dtype, Self.target
        ]() else None
    )
