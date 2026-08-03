# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests the HAL `Context` loaded-function cache."""

from std.os import getenv
from _hal import Context, Driver
from _hal.device import Device, DeviceSpec, get_device_spec
from std.gpu import global_idx
from std.memory import ArcPointer, UnsafePointer
from std.testing import assert_equal, assert_not_equal


def scale_kernel(data: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    var tid = global_idx.x
    if tid < n:
        data[unsafe_offset=tid] = data[unsafe_offset=tid] * 2.0


def test_content_dedupes_across_buffers[
    device_spec: DeviceSpec,
](mut ctx: Context[device_spec]) raises:
    """Identical code in two distinct buffers loads once."""
    var info = ctx._compile_inner[type_of(scale_kernel), scale_kernel]()
    var name = String(info.function_name)

    # Append rather than `String(info.asm)`, which only views the static buffer;
    # two views would share an address and an address key would "dedupe" them,
    # so the assert below guards against a vacuous pass.
    var copy_a = String()
    copy_a += info.asm
    var copy_b = String()
    copy_b += info.asm
    assert_not_equal(Int(copy_a.unsafe_ptr()), Int(copy_b.unsafe_ptr()))

    var func_a = ctx.get_or_load_function_dynamic(
        StringSlice(copy_a), name.copy()
    )
    var func_b = ctx.get_or_load_function_dynamic(
        StringSlice(copy_b), name.copy()
    )
    assert_equal(Int(func_a), Int(func_b))


def test_static_and_dynamic_agree[
    device_spec: DeviceSpec,
](mut ctx: Context[device_spec]) raises:
    """The address fast path resolves to the same entry as a content lookup."""
    var info = ctx._compile_inner[type_of(scale_kernel), scale_kernel]()
    var name = String(info.function_name)

    var first = ctx.get_or_load_function(info.asm, name.copy())
    var second = ctx.get_or_load_function(info.asm, name.copy())
    assert_equal(Int(first), Int(second))

    var copied = String()
    copied += info.asm
    var dynamic = ctx.get_or_load_function_dynamic(
        StringSlice(copied), name.copy()
    )
    assert_equal(Int(first), Int(dynamic))


def test_cache_key_distinguishes[device_spec: DeviceSpec]() raises:
    """Every component of the key participates, so nothing aliases on a partial
    match."""
    comptime Ctx = Context[device_spec]
    var base = Ctx._function_cache_key(1, 10, "kernel", None, 0)

    assert_not_equal(base, Ctx._function_cache_key(2, 10, "kernel", None, 0))
    assert_not_equal(base, Ctx._function_cache_key(1, 11, "kernel", None, 0))
    assert_not_equal(base, Ctx._function_cache_key(1, 10, "other", None, 0))
    assert_not_equal(
        base, Ctx._function_cache_key(1, 10, "kernel", Int32(8), 0)
    )
    assert_not_equal(
        Ctx._function_cache_key(1, 10, "kernel", Int32(8), 0),
        Ctx._function_cache_key(1, 10, "kernel", Int32(8), 1),
    )
    assert_equal(base, Ctx._function_cache_key(1, 10, "kernel", None, 0))


def main() raises:
    var plugin_spec = getenv("MODULAR_DRIVER_PLUGINS")
    if not plugin_spec:
        raise Error("MODULAR_DRIVER_PLUGINS not set")

    comptime device_spec = get_device_spec[0]()
    test_cache_key_distinguishes[device_spec]()

    var driver = Driver.create(plugin_spec)
    var device: ArcPointer[Device[device_spec]] = driver[].get_device[0]()
    var ctx: ArcPointer[Context[device_spec]] = device[].get_context()

    test_content_dedupes_across_buffers(ctx[])
    test_static_and_dynamic_agree(ctx[])
    print("Function cache tests OK")
