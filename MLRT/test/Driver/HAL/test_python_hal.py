# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Lifecycle smoke tests for max.sys._hal.

Exercises the Python -> Mojo HAL FFI path against the real HAL plugin
selected by the build.
"""

from __future__ import annotations

import ctypes

import mojo.importer  # noqa: F401  # registers Mojo .so importer hook
from max.sys._hal import Buffer, Bundle, Context, Driver, Event, Function
from vec_add import vec_add  # type: ignore[import-not-found]

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_host_buf(
    data: bytes | bytearray,
) -> tuple[ctypes.Array[ctypes.c_uint8], int]:
    n = len(data)
    buf = (ctypes.c_uint8 * n)(*data)
    return buf, ctypes.addressof(buf)


def _make_empty_host_buf(
    n: int,
) -> tuple[ctypes.Array[ctypes.c_uint8], int]:
    buf = (ctypes.c_uint8 * n)()
    return buf, ctypes.addressof(buf)


def _bytes_of(buf: ctypes.Array[ctypes.c_uint8]) -> bytes:
    return bytes(buf)


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


def test_driver_load() -> None:
    driver = Driver()
    assert isinstance(driver.name, str) and driver.name != ""
    assert driver.device_count >= 1


def test_queue_record_event_and_synchronize() -> None:
    ctx = Driver().get_device(0).get_context()
    queue = ctx.create_queue()
    event = queue.record_event()
    assert isinstance(event, Event)
    event.synchronize()
    assert event.is_ready() is True


# ----------------------------------------------------------------------
# Buffer / memory allocation
# ----------------------------------------------------------------------


def test_context_alloc_sync_returns_buffer() -> None:
    ctx = Driver().get_device(0).get_context()
    buf = ctx.alloc_sync(64)
    assert isinstance(buf, Buffer)
    assert buf.byte_size == 64
    assert ctx.memory_get_address(buf) != 0


def test_context_alloc_host_pinned_returns_buffer() -> None:
    ctx = Driver().get_device(0).get_context()
    buf = ctx.alloc_host_pinned(64)
    assert isinstance(buf, Buffer)
    assert buf.byte_size == 64


# ----------------------------------------------------------------------
# Memory copies via Queue
# ----------------------------------------------------------------------


def test_queue_host_to_device_to_host_roundtrip() -> None:
    ctx = Driver().get_device(0).get_context()
    queue = ctx.create_queue()

    n = 256
    src_data = bytes(range(n))
    _src, src_addr = _make_host_buf(src_data)
    dst, dst_addr = _make_empty_host_buf(n)
    dev = ctx.alloc_sync(n)

    queue.copy_to_device(dev, src_addr, n)
    queue.copy_from_device(dst_addr, dev, n)
    queue.synchronize()

    assert _bytes_of(dst) == src_data


def test_queue_copy_intra_device() -> None:
    ctx = Driver().get_device(0).get_context()
    queue = ctx.create_queue()

    n = 256
    src_data = bytes((i * 3) % 256 for i in range(n))
    _src, src_addr = _make_host_buf(src_data)
    dst, dst_addr = _make_empty_host_buf(n)
    dev_a = ctx.alloc_sync(n)
    dev_b = ctx.alloc_sync(n)

    queue.copy_to_device(dev_a, src_addr, n)
    queue.copy_intra_device(dev_b, dev_a, n)
    queue.copy_from_device(dst_addr, dev_b, n)
    queue.synchronize()

    assert _bytes_of(dst) == src_data


# ----------------------------------------------------------------------
# Memory set / fill via Queue
# ----------------------------------------------------------------------


def test_queue_set_memory() -> None:
    ctx = Driver().get_device(0).get_context()
    queue = ctx.create_queue()

    n = 128
    dst, dst_addr = _make_empty_host_buf(n)
    dev = ctx.alloc_sync(n)

    queue.set_memory(dev, 0x5A, n)
    queue.copy_from_device(dst_addr, dev, n)
    queue.synchronize()

    assert _bytes_of(dst) == bytes([0x5A]) * n


def test_queue_fill() -> None:
    ctx = Driver().get_device(0).get_context()
    queue = ctx.create_queue()

    n = 128
    dst, dst_addr = _make_empty_host_buf(n)
    dev = ctx.alloc_sync(n)

    # 32-bit pattern, little-endian repeating across the buffer.
    queue.fill(dev, 0xDEADBEEF, 4, n)
    queue.copy_from_device(dst_addr, dev, n)
    queue.synchronize()

    assert _bytes_of(dst) == (0xDEADBEEF).to_bytes(4, "little") * (n // 4)


# ----------------------------------------------------------------------
# Memory copies via Stream
# ----------------------------------------------------------------------


def test_stream_host_to_device_to_host_roundtrip() -> None:
    ctx = Driver().get_device(0).get_context()
    stream = ctx.create_stream()

    n = 256
    src_data = bytes(range(n))
    _src, src_addr = _make_host_buf(src_data)
    dst, dst_addr = _make_empty_host_buf(n)
    dev = ctx.alloc_sync(n)

    stream.copy_to_device(dev, src_addr, n)
    stream.copy_from_device(dst_addr, dev, n)
    stream.synchronize()

    assert _bytes_of(dst) == src_data


def test_stream_copy_intra_device() -> None:
    ctx = Driver().get_device(0).get_context()
    stream = ctx.create_stream()

    n = 256
    src_data = bytes((i * 7) % 256 for i in range(n))
    _src, src_addr = _make_host_buf(src_data)
    dst, dst_addr = _make_empty_host_buf(n)
    dev_a = ctx.alloc_sync(n)
    dev_b = ctx.alloc_sync(n)

    stream.copy_to_device(dev_a, src_addr, n)
    stream.copy_intra_device(dev_b, dev_a, n)
    stream.copy_from_device(dst_addr, dev_b, n)
    stream.synchronize()

    assert _bytes_of(dst) == src_data


def test_stream_set_memory() -> None:
    ctx = Driver().get_device(0).get_context()
    stream = ctx.create_stream()

    n = 128
    dst, dst_addr = _make_empty_host_buf(n)
    dev = ctx.alloc_sync(n)

    stream.set_memory(dev, 0x5A, n)
    stream.copy_from_device(dst_addr, dev, n)
    stream.synchronize()

    assert _bytes_of(dst) == bytes([0x5A]) * n


def test_stream_fill() -> None:
    ctx = Driver().get_device(0).get_context()
    stream = ctx.create_stream()

    n = 128
    dst, dst_addr = _make_empty_host_buf(n)
    dev = ctx.alloc_sync(n)

    stream.fill(dev, 0xDEADBEEF, 4, n)
    stream.copy_from_device(dst_addr, dev, n)
    stream.synchronize()

    assert _bytes_of(dst) == (0xDEADBEEF).to_bytes(4, "little") * (n // 4)


# ----------------------------------------------------------------------
# Cross-queue / cross-stream wait_for_events
# ----------------------------------------------------------------------


def test_queue_wait_for_events_single() -> None:
    ctx = Driver().get_device(0).get_context()
    producer = ctx.create_queue()
    consumer = ctx.create_queue()
    event = producer.record_event()
    consumer.wait_for_events(event)
    consumer.synchronize()


def test_stream_wait_for_events_single() -> None:
    ctx = Driver().get_device(0).get_context()
    producer = ctx.create_stream()
    consumer = ctx.create_stream()
    event = producer.record_event()
    consumer.wait_for_events(event)
    consumer.synchronize()


# ----------------------------------------------------------------------
# Mojo kernel -> Python compile path
# ----------------------------------------------------------------------


def test_compile_returns_bundle_with_name() -> None:
    ctx = Driver().get_device(0).get_context()
    bundle = ctx.compile(vec_add)
    assert isinstance(bundle, Bundle)
    assert isinstance(bundle.function_name, str)
    assert len(bundle.function_name) > 0


def test_load_function_returns_function() -> None:
    ctx = Driver().get_device(0).get_context()
    bundle = ctx.compile(vec_add)
    func = ctx.load_function(bundle, bundle.function_name)
    assert isinstance(func, Function)


# ----------------------------------------------------------------------
# Kernel launch via Queue / Stream
# ----------------------------------------------------------------------


def _expected_vec_add(i: int) -> float:
    return float((i % 2) * -i * 2 + i)


def _prepare_vec_add_inputs(
    ctx: Context,
    length: int,
) -> tuple[
    Buffer,
    Buffer,
    Buffer,
    ctypes.Array[ctypes.c_float],
    ctypes.Array[ctypes.c_float],
    int,
]:
    buf_size = length * ctypes.sizeof(ctypes.c_float)
    a_host = (ctypes.c_float * length)(*(float(i) for i in range(length)))
    b_host = (ctypes.c_float * length)(
        *(float((i % 2) * -i * 2) for i in range(length))
    )
    a_dev = ctx.alloc_sync(buf_size)
    b_dev = ctx.alloc_sync(buf_size)
    out_dev = ctx.alloc_sync(buf_size)
    return a_dev, b_dev, out_dev, a_host, b_host, buf_size


def _vec_add_kernel_args(
    ctx: Context,
    a_dev: Buffer,
    b_dev: Buffer,
    out_dev: Buffer,
    length: int,
) -> tuple[list[int], list[int], list[object]]:
    a_ptr = ctypes.c_uint64(ctx.memory_get_address(a_dev))
    b_ptr = ctypes.c_uint64(ctx.memory_get_address(b_dev))
    out_ptr = ctypes.c_uint64(ctx.memory_get_address(out_dev))
    n_val = ctypes.c_int64(length)
    addrs = [
        ctypes.addressof(a_ptr),
        ctypes.addressof(b_ptr),
        ctypes.addressof(out_ptr),
        ctypes.addressof(n_val),
    ]
    sizes = [
        ctypes.sizeof(a_ptr),
        ctypes.sizeof(b_ptr),
        ctypes.sizeof(out_ptr),
        ctypes.sizeof(n_val),
    ]
    return addrs, sizes, [a_ptr, b_ptr, out_ptr, n_val]


def test_queue_execute_vec_add() -> None:
    ctx = Driver().get_device(0).get_context()
    bundle = ctx.compile(vec_add)
    func = ctx.load_function(bundle, bundle.function_name)
    queue = ctx.create_queue()

    length = 1024
    a_dev, b_dev, out_dev, a_host, b_host, buf_size = _prepare_vec_add_inputs(
        ctx, length
    )
    out_host = (ctypes.c_float * length)()

    queue.copy_to_device(a_dev, ctypes.addressof(a_host), buf_size)
    queue.copy_to_device(b_dev, ctypes.addressof(b_host), buf_size)
    queue.synchronize()

    args, arg_sizes, _keepalive = _vec_add_kernel_args(
        ctx, a_dev, b_dev, out_dev, length
    )

    block_dim = 256
    grid_dim = length // block_dim
    queue.execute(
        func,
        grid=(grid_dim, 1, 1),
        block=(block_dim, 1, 1),
        args=args,
        arg_sizes=arg_sizes,
    )
    queue.synchronize()

    queue.copy_from_device(ctypes.addressof(out_host), out_dev, buf_size)
    queue.synchronize()

    for i in range(length):
        assert out_host[i] == _expected_vec_add(i), (
            f"out_host[{i}] = {out_host[i]} != {_expected_vec_add(i)}"
        )


def test_stream_execute_vec_add() -> None:
    ctx = Driver().get_device(0).get_context()
    bundle = ctx.compile(vec_add)
    func = ctx.load_function(bundle, bundle.function_name)
    stream = ctx.create_stream()

    length = 1024
    a_dev, b_dev, out_dev, a_host, b_host, buf_size = _prepare_vec_add_inputs(
        ctx, length
    )
    out_host = (ctypes.c_float * length)()

    stream.copy_to_device(a_dev, ctypes.addressof(a_host), buf_size)
    stream.copy_to_device(b_dev, ctypes.addressof(b_host), buf_size)

    args, arg_sizes, _keepalive = _vec_add_kernel_args(
        ctx, a_dev, b_dev, out_dev, length
    )

    block_dim = 256
    grid_dim = length // block_dim
    stream.execute(
        func,
        grid=(grid_dim, 1, 1),
        block=(block_dim, 1, 1),
        args=args,
        arg_sizes=arg_sizes,
    )

    stream.copy_from_device(ctypes.addressof(out_host), out_dev, buf_size)
    stream.synchronize()

    for i in range(length):
        assert out_host[i] == _expected_vec_add(i), (
            f"out_host[{i}] = {out_host[i]} != {_expected_vec_add(i)}"
        )


def test_queue_execute_validates_shapes() -> None:
    import pytest

    ctx = Driver().get_device(0).get_context()
    bundle = ctx.compile(vec_add)
    func = ctx.load_function(bundle, bundle.function_name)
    queue = ctx.create_queue()

    with pytest.raises(ValueError, match="grid"):
        queue.execute(func, grid=(1, 1), block=(1, 1, 1), args=[], arg_sizes=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="block"):
        queue.execute(func, grid=(1, 1, 1), block=(1, 1), args=[], arg_sizes=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="same length"):
        queue.execute(
            func, grid=(1, 1, 1), block=(1, 1, 1), args=[0], arg_sizes=[]
        )
