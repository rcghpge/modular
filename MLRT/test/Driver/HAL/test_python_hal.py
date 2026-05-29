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
from max.sys._hal import Buffer, Bundle, Driver, Event, Function
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
