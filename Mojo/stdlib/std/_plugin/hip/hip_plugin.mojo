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
    _GlobalPerfCounterNsPluginHookFnType,
    _PrintEmitPluginHookFnType,
    _SleepPluginHookFnType,
    _UnaryMathPluginHookFnType,
)
from std.collections import OptionalReg
from std.sys.info import CompilationTarget
from std.collections.string.string_span import (
    _get_kgen_string,
    get_static_string,
)
from std.ffi import CStringSpan
from std.io.file_descriptor import RawFileDescriptor
from std.reflection.location import SourceLocation
from std.sys._amdgpu import (
    printf_append_args,
    printf_append_string_n,
    printf_begin,
)

from .hip_math import _amdgcn_unary_math_fn
from .hip_time import _global_perf_counter_ns, _sleep


def _print_emit[O: Origin](cstr: CStringSpan[O], file: RawFileDescriptor):
    """Writes `print()` output from an AMD kernel through the hostcall
    fprintf service.

    The service writes to the host's stdout, so `file` is ignored.
    """
    var msg = printf_begin()
    _ = printf_append_string_n(msg, cstr.as_bytes_with_nul(), is_last=True)


# TODO(MSTDL-1783): Use `_printf` once it passes runtime `%s` arguments on
# AMD GPUs.
@inline(.always)
def _debug_assert_emit[
    O: Origin
](message: Pointer[Byte, O], length: Int, loc: SourceLocation):
    """Writes a failed `debug_assert` message, with the failing block and
    thread, through the hostcall fprintf service.

    The format string, file name, integer arguments, and message are appended
    to the fprintf message one piece at a time.
    """
    # The compile-time interpreter cannot run the hostcall.
    if __is_run_in_comptime_interpreter:
        print("At: ", loc, ": Assert Error: ", message, sep="")
        return

    from std._gpu.primitives.id import block_idx, thread_idx

    comptime fmt = (
        "At: %s:%llu:%llu: block: [%llu,%llu,%llu] thread: [%llu,%llu,%llu]"
        " Assert Error: %s\n"
    )

    var fd = printf_begin()
    # Each appended string must carry its own nul terminator so the AMD
    # fprintf service can delimit it; `as_bytes()` omits the nul, which
    # corrupts output when a string's length is a multiple of 8.
    # `get_static_string` guarantees a trailing nul in static memory
    # just past the returned range.
    var fmt_str = get_static_string[fmt]()
    _ = printf_append_string_n(
        fd,
        Span(
            unsafe_ptr=fmt_str.as_bytes().unsafe_ptr(),
            length=fmt_str.byte_length() + 1,
        ),
        False,
    )
    # `file_name()` is a string literal, so its trailing nul lives in static
    # memory just past the range.
    var file_name = loc.file_name()
    _ = printf_append_string_n(
        fd,
        Span(
            unsafe_ptr=file_name.as_bytes().unsafe_ptr(),
            length=file_name.byte_length() + 1,
        ),
        False,
    )
    # The service takes at most 7 integer arguments per append.
    _ = printf_append_args(
        fd,
        7,
        UInt64(loc.line()),
        UInt64(loc.column()),
        UInt64(block_idx.x),
        UInt64(block_idx.y),
        UInt64(block_idx.z),
        UInt64(thread_idx.x),
        UInt64(thread_idx.y),
        0,
    )
    _ = printf_append_args(fd, 1, UInt64(thread_idx.z), 0, 0, 0, 0, 0, 0, 0)
    _ = printf_append_string_n(
        fd, Span(unsafe_ptr=message, length=length), True
    )


struct HIPPlugin(StdPlugin):
    """`PluginHooks` implementation for AMD HIP backends."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["hip"]()
    comptime HookFor[target: CompilationTarget] = HIPPluginHooks[target]


struct HIPPluginHooks[target: CompilationTarget](PluginHooks):
    comptime print_emit_fn: OptionalReg[
        _PrintEmitPluginHookFnType
    ] = _print_emit

    comptime global_perf_counter_ns_fn: OptionalReg[
        _GlobalPerfCounterNsPluginHookFnType
    ] = _global_perf_counter_ns

    comptime sleep_fn: OptionalReg[_SleepPluginHookFnType] = _sleep

    @staticmethod
    @inline(.always)
    def debug_assert_emit_fn[
        O: Origin
    ](message: Pointer[Byte, O], length: Int, loc: SourceLocation):
        _debug_assert_emit(message, length, loc)

    comptime _handles_debug_assert: Bool = True

    comptime unary_math_fn[
        name: StaticString, dtype: DType, width: Int
    ]: OptionalReg[_UnaryMathPluginHookFnType] = _amdgcn_unary_math_fn[
        name, dtype
    ]()
