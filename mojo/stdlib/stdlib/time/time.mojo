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
"""Implements basic utils for working with time.

You can import these APIs from the `time` package. For example:

```mojo
from time import perf_counter_ns
```
"""

from math import floor
from os import abort
from sys import (
    CompilationTarget,
    external_call,
    is_amd_gpu,
    is_gpu,
    is_nvidia_gpu,
    llvm_intrinsic,
)


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#

# Enums used in time.h 's glibc
alias _CLOCK_REALTIME = 0
alias _CLOCK_MONOTONIC = 1 if CompilationTarget.is_linux() else 6
alias _CLOCK_PROCESS_CPUTIME_ID = 2 if CompilationTarget.is_linux() else 12
alias _CLOCK_THREAD_CPUTIME_ID = 3 if CompilationTarget.is_linux() else 16
alias _CLOCK_MONOTONIC_RAW = 4

# Constants
alias _NSEC_PER_USEC = 1000
alias _NSEC_PER_MSEC = 1_000_000
alias _USEC_PER_MSEC = 1000
alias _MSEC_PER_SEC = 1000
alias _NSEC_PER_SEC = _NSEC_PER_USEC * _USEC_PER_MSEC * _MSEC_PER_SEC

# LARGE_INTEGER in Windows represent a signed 64 bit integer. Internally it
# is implemented as a union of one 64 bit integer or two 32 bit integers
# for 64/32 bit compilers.
# https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-large_integer-r1
alias _WINDOWS_LARGE_INTEGER = Int64


@fieldwise_init
@register_passable("trivial")
struct _CTimeSpec(Copyable, Defaultable, Movable, Stringable, Writable):
    var tv_sec: Int  # Seconds
    var tv_subsec: Int  # subsecond (nanoseconds on linux and usec on mac)

    fn __init__(out self):
        self.tv_sec = 0
        self.tv_subsec = 0

    fn as_nanoseconds(self) -> UInt:
        @parameter
        if CompilationTarget.is_linux():
            return self.tv_sec * _NSEC_PER_SEC + self.tv_subsec
        else:
            return self.tv_sec * _NSEC_PER_SEC + self.tv_subsec * _NSEC_PER_USEC

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.as_nanoseconds(), "ns")


@fieldwise_init
@register_passable("trivial")
struct _FILETIME(Copyable, Defaultable, Movable):
    var dw_low_date_time: UInt32
    var dw_high_date_time: UInt32

    fn __init__(out self):
        self.dw_low_date_time = 0
        self.dw_high_date_time = 0

    fn as_nanoseconds(self) -> UInt:
        # AFTER subtracting windows offset the return value fits in a signed int64
        # BEFORE subtracting windows offset the return value does not fit in a signed int64
        # Taken from https://github.com/microsoft/STL/blob/c8d1efb6d504f6392acf8f6d01fd703f7c8826c0/stl/src/xtime.cpp#L50
        alias windows_to_unix_epoch_offset_ns: Int = 0x19DB1DED53E8000
        var interval_count: UInt64 = (
            (self.dw_high_date_time.cast[DType.uint64]() << 32)
            + self.dw_low_date_time.cast[DType.uint64]()
            - windows_to_unix_epoch_offset_ns
        )
        return Int(interval_count * 100)


@always_inline
fn _clock_gettime(clockid: Int) -> _CTimeSpec:
    """Low-level call to the clock_gettime libc function"""
    var ts = _CTimeSpec()

    # Call libc's clock_gettime.
    _ = external_call["clock_gettime", Int32](Int32(clockid), Pointer(to=ts))

    return ts


@always_inline
fn _gettime_as_nsec_unix(clockid: Int) -> UInt:
    if CompilationTarget.is_linux():
        var ts = _clock_gettime(clockid)
        return ts.as_nanoseconds()
    else:
        return Int(
            external_call["clock_gettime_nsec_np", Int64](Int32(clockid))
        )


@always_inline
fn _gpu_clock() -> UInt:
    """Returns a 64-bit unsigned cycle counter."""
    alias asm = _gpu_clock_inst()
    return Int(llvm_intrinsic[asm, Int64]())


fn _gpu_clock_inst() -> StaticString:
    @parameter
    if is_nvidia_gpu():
        return "llvm.nvvm.read.ptx.sreg.clock64"
    elif is_amd_gpu():
        return "llvm.amdgcn.s.memtime"
    else:
        return CompilationTarget.unsupported_target_error[
            StaticString,
            operation="_gpu_clock",
        ]()


@always_inline
fn _realtime_nanoseconds() -> UInt:
    """Returns the current realtime time in nanoseconds"""
    return _gettime_as_nsec_unix(_CLOCK_REALTIME)


@always_inline
fn _monotonic_nanoseconds() -> UInt:
    """Returns the current monotonic time in nanoseconds"""

    @parameter
    if is_gpu():
        return _gpu_clock()
    elif CompilationTarget.is_windows():
        var ft = _FILETIME()
        external_call["GetSystemTimePreciseAsFileTime", NoneType](
            Pointer(to=ft)
        )

        return ft.as_nanoseconds()
    else:
        return _gettime_as_nsec_unix(_CLOCK_MONOTONIC)


@always_inline
fn _monotonic_raw_nanoseconds() -> UInt:
    """Returns the current monotonic time in nanoseconds"""
    return _gettime_as_nsec_unix(_CLOCK_MONOTONIC_RAW)


@always_inline
fn _process_cputime_nanoseconds() -> UInt:
    """Returns the high-resolution per-process timer from the CPU"""

    return _gettime_as_nsec_unix(_CLOCK_PROCESS_CPUTIME_ID)


@always_inline
fn _thread_cputime_nanoseconds() -> UInt:
    """Returns the thread-specific CPU-time clock"""

    return _gettime_as_nsec_unix(_CLOCK_THREAD_CPUTIME_ID)


# ===-----------------------------------------------------------------------===#
# perf_counter
# ===-----------------------------------------------------------------------===#


@always_inline
fn perf_counter() -> Float64:
    """Return the value (in fractional seconds) of a performance counter, i.e.
    a clock with the highest available resolution to measure a short duration.
    It does include time elapsed during sleep and is system-wide. The reference
    point of the returned value is undefined, so that only the difference
    between the results of two calls is valid.

    Returns:
        The current time in ns.
    """
    return Float64(_monotonic_nanoseconds()) / _NSEC_PER_SEC


# ===-----------------------------------------------------------------------===#
# perf_counter_ns
# ===-----------------------------------------------------------------------===#


@always_inline
fn perf_counter_ns() -> UInt:
    """Return the value (in nanoseconds) of a performance counter, i.e.
    a clock with the highest available resolution to measure a short duration.
    It does include time elapsed during sleep and is system-wide. The reference
    point of the returned value is undefined, so that only the difference
    between the results of two calls is valid.

    Returns:
        The current time in ns.
    """
    return _monotonic_nanoseconds()


# ===-----------------------------------------------------------------------===#
# monotonic
# ===-----------------------------------------------------------------------===#


@always_inline
fn monotonic() -> UInt:
    """
    Returns the current monotonic time time in nanoseconds. This function
    queries the current platform's monotonic clock, making it useful for
    measuring time differences, but the significance of the returned value
    varies depending on the underlying implementation.

    Returns:
        The current time in ns.
    """
    return perf_counter_ns()


# ===-----------------------------------------------------------------------===#
# time_function
# ===-----------------------------------------------------------------------===#


@always_inline
@parameter
fn _time_function_windows[
    func: fn () raises capturing [_] -> None
]() raises -> UInt:
    """Calculates elapsed time in Windows"""

    var ticks_per_sec: _WINDOWS_LARGE_INTEGER = 0
    var ticks_per_sec_ptr = UnsafePointer[_WINDOWS_LARGE_INTEGER](
        to=ticks_per_sec
    )
    external_call["QueryPerformanceFrequency", NoneType](ticks_per_sec_ptr)

    var starting_tick_count: _WINDOWS_LARGE_INTEGER = 0
    var start_ptr = UnsafePointer[_WINDOWS_LARGE_INTEGER](
        to=starting_tick_count
    )
    var ending_tick_count: _WINDOWS_LARGE_INTEGER = 0
    var end_ptr = UnsafePointer[_WINDOWS_LARGE_INTEGER](to=ending_tick_count)

    external_call["QueryPerformanceCounter", NoneType](start_ptr)
    func()
    external_call["QueryPerformanceCounter", NoneType](end_ptr)

    var elapsed_ticks = ending_tick_count - starting_tick_count

    # Note: Windows performance counter resolution is in µs.
    var elapsed_time_in_ns = (elapsed_ticks * 1_000_000_000) // ticks_per_sec
    return Int(elapsed_time_in_ns)


@always_inline
@parameter
fn time_function[func: fn () raises capturing [_] -> None]() raises -> UInt:
    """Measures the time spent in the function.

    Parameters:
        func: The function to time.

    Returns:
        The time elapsed in the function in ns.
    """

    @parameter
    if CompilationTarget.is_windows():
        return _time_function_windows[func]()

    var tic = perf_counter_ns()
    func()
    var toc = perf_counter_ns()
    return toc - tic


@always_inline
@parameter
fn time_function[func: fn () capturing [_] -> None]() -> UInt:
    """Measures the time spent in the function.

    Parameters:
        func: The function to time.

    Returns:
        The time elapsed in the function in ns.
    """

    @parameter
    fn raising_func() raises:
        func()

    try:
        return time_function[raising_func]()
    except err:
        return abort[UInt](String(err))


# ===-----------------------------------------------------------------------===#
# sleep
# ===-----------------------------------------------------------------------===#


fn sleep(sec: Float64):
    """Suspends the current thread for the seconds specified.

    Args:
        sec: The number of seconds to sleep for.
    """

    @parameter
    if is_gpu():
        var nsec = sec * 1.0e9
        alias intrinsic = _gpu_sleep_inst()
        llvm_intrinsic[intrinsic, NoneType](nsec.cast[DType.int32]())
        return

    alias NANOSECONDS_IN_SECOND = 1_000_000_000
    var total_secs = floor(sec)
    var tv_spec = _CTimeSpec(
        Int(total_secs),
        Int((sec - total_secs) * NANOSECONDS_IN_SECOND),
    )
    var req = UnsafePointer[_CTimeSpec](to=tv_spec)
    var rem = UnsafePointer[_CTimeSpec]()
    _ = external_call["nanosleep", Int32](req, rem)
    _ = tv_spec
    _ = req
    _ = rem


fn _gpu_sleep_inst() -> StaticString:
    @parameter
    if is_nvidia_gpu():
        return "llvm.nvvm.nanosleep"
    elif is_amd_gpu():
        return "llvm.amdgcn.s.sleep"
    else:
        return CompilationTarget.unsupported_target_error[
            StaticString,
            operation="sleep",
        ]()


fn sleep(sec: UInt):
    """Suspends the current thread for the seconds specified.

    Args:
        sec: The number of seconds to sleep for.
    """

    @parameter
    if is_gpu():
        return sleep(Float64(sec))

    @parameter
    if CompilationTarget.is_windows():
        # In Windows the argument is in milliseconds.
        external_call["Sleep", NoneType](sec * 1000)
    else:
        external_call["sleep", NoneType](Int32(sec))
