# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from std.os import getenv
from _hal import Driver, Context
from _hal.device import Device, DeviceSpec, get_device_spec
from _hal.status import HALError
from _hal.plugin import MemoryHandle
from std.gpu import global_idx
from std.memory import alloc, ArcPointer, OpaquePointer, UnsafePointer
from std.sys import size_of
from std.testing import assert_equal


def vec_add(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    result: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
):
    var tid = global_idx.x
    if tid >= n:
        return
    result[tid] = a[tid] + b[tid]


def do_execute[
    device_spec: DeviceSpec,
](mut ctx: Context[device_spec]) raises:
    # Compile the kernel and load it into the plugin.
    ref (bundle, info) = ctx.compile[type_of(vec_add), vec_add]()
    print("Compiled and loaded bundle")
    print("  function_name:", info.function_name)

    # Load function from the bundle.
    var func = ctx.load_function(bundle, info.function_name)
    print("Loaded function handle")

    # Allocate device memory.
    comptime length = 1024
    comptime elem_size = size_of[Float32]()
    comptime buf_size = length * elem_size

    var in0_dev = ctx.alloc_sync(UInt64(buf_size))
    var in1_dev = ctx.alloc_sync(UInt64(buf_size))
    var out_dev = ctx.alloc_sync(UInt64(buf_size))

    # Create queue.
    var queue = ctx.create_queue()

    # Fill host buffers and copy to device.
    var in0_host = alloc[Float32](length)
    var in1_host = alloc[Float32](length)
    var out_host = alloc[Float32](length)
    for i in range(length):
        in0_host[i] = Float32(i)
        in1_host[i] = Float32((i % 2) * -i * 2)
        out_host[i] = Float32(0.0)

    queue[].copy_to_device(
        in0_dev.view(),
        rebind[UnsafePointer[UInt8, MutAnyOrigin]](in0_host),
    )
    queue[].copy_to_device(
        in1_dev.view(),
        rebind[UnsafePointer[UInt8, MutAnyOrigin]](in1_host),
    )
    queue[].synchronize()

    # Get GPU addresses for each buffer.
    var in0_ptr = ctx.memory_get_address(in0_dev)
    var in1_ptr = ctx.memory_get_address(in1_dev)
    var out_ptr = ctx.memory_get_address(out_dev)
    var length_val = Int(length)

    # Pack kernel arguments — order must match the kernel signature: a, b, result, n.
    var args = alloc[OpaquePointer[MutUntrackedOrigin]](4)
    var arg_sizes = alloc[UInt64](4)

    args[0] = rebind[OpaquePointer[MutUntrackedOrigin]](
        UnsafePointer(to=in0_ptr)
    )
    args[1] = rebind[OpaquePointer[MutUntrackedOrigin]](
        UnsafePointer(to=in1_ptr)
    )
    args[2] = rebind[OpaquePointer[MutUntrackedOrigin]](
        UnsafePointer(to=out_ptr)
    )
    args[3] = rebind[OpaquePointer[MutUntrackedOrigin]](
        UnsafePointer(to=length_val)
    )
    arg_sizes[0] = UInt64(size_of[UInt64]())
    arg_sizes[1] = UInt64(size_of[UInt64]())
    arg_sizes[2] = UInt64(size_of[UInt64]())
    arg_sizes[3] = UInt64(size_of[Int]())

    # Execute.
    comptime block_dim = UInt32(256)
    comptime grid_dim = UInt32(length // Int(block_dim))

    queue[].execute(
        func,
        grid=(grid_dim, UInt32(1), UInt32(1)),
        block=(block_dim, UInt32(1), UInt32(1)),
        args=rebind[
            UnsafePointer[OpaquePointer[MutUntrackedOrigin], MutAnyOrigin]
        ](args),
        arg_sizes=rebind[UnsafePointer[UInt64, MutAnyOrigin]](arg_sizes),
        num_args=UInt32(4),
    )

    # Copy results back.
    queue[].copy_from_device(
        rebind[UnsafePointer[UInt8, MutAnyOrigin]](out_host),
        out_dev.view(),
    )
    queue[].synchronize()

    # Verify.
    for i in range(length):
        var fi = Float32(i)
        var expected = (fi % 2) * -fi * 2 + fi
        assert_equal(out_host[i], expected)

    # Also test out DtoD copies
    var out_dev_copy = ctx.alloc_sync(UInt64(buf_size))
    var out_host_copy = alloc[Float32](length)
    for i in range(length):
        out_host_copy[i] = Float32(0.0)

    queue[].copy_intra_device(out_dev_copy.view(), out_dev.view())
    queue[].synchronize()
    queue[].copy_from_device(
        rebind[UnsafePointer[UInt8, MutAnyOrigin]](out_host_copy),
        out_dev_copy.view(),
    )
    queue[].synchronize()

    for i in range(length):
        assert_equal(out_host_copy[i], out_host[i])

    # Cleanup.
    out_host_copy.free()
    args.free()
    arg_sizes.free()
    in0_host.free()
    in1_host.free()
    out_host.free()
    ctx.unload_function(func)
    ctx.free_sync(out_dev_copy^)
    ctx.free_sync(out_dev^)
    ctx.free_sync(in1_dev^)
    ctx.free_sync(in0_dev^)


def do_offset_copy_test[
    device_spec: DeviceSpec,
](mut ctx: Context[device_spec]) raises:
    """Copies into the upper half of a buffer through an offset `BufferView`
    and reads the whole buffer back."""
    comptime n = 256
    comptime half = n // 2

    var stream = ctx.create_stream()
    var device_buffer = ctx.alloc_sync(UInt64(n))

    var src = alloc[UInt8](half)
    for i in range(half):
        src[i] = UInt8(0xAB)

    # Fill the lower half, then write the upper half at byte offset `half`.
    stream[].set_memory(
        device_buffer.view(byte_offset=UInt64(0), byte_size=UInt64(half)),
        value=UInt64(0xCD),
        value_size=UInt64(1),
    )
    stream[].copy_to_device(
        device_buffer.view(byte_offset=UInt64(half), byte_size=UInt64(half)),
        rebind[UnsafePointer[UInt8, MutAnyOrigin]](src),
    )

    var host = alloc[UInt8](n)
    stream[].copy_from_device(
        rebind[UnsafePointer[UInt8, MutAnyOrigin]](host), device_buffer.view()
    )
    stream[].synchronize()
    for i in range(n):
        assert_equal(host[i], UInt8(0xCD) if i < half else UInt8(0xAB))

    src.free()
    host.free()
    ctx.free_sync(device_buffer^)


def main() raises:
    var plugin_spec = getenv("MODULAR_DRIVER_PLUGINS")
    if not plugin_spec:
        raise Error("MODULAR_DRIVER_PLUGINS not set")

    var driver: ArcPointer[Driver]
    try:
        driver = Driver.create(plugin_spec)
    except e:
        raise Error(String.write(e))
    print("Driver:", driver[].get_name())
    print("Devices:", driver[].get_device_count())

    var device: ArcPointer[Device[get_device_spec[0]()]]
    try:
        device = driver[].get_device[0]()
    except e:
        raise Error(String.write(e))
    print("Device", device[].id, "OK")

    var ctx: ArcPointer[Context[get_device_spec[0]()]]
    try:
        ctx = device[].get_context()
    except e:
        raise Error(String.write(e))
    print("Context created")

    do_execute(ctx[])
    do_offset_copy_test(ctx[])
