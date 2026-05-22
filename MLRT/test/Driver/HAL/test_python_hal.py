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

from max.sys._hal import Context, Device, Driver, Event, Queue, Stream


def test_driver_load() -> None:
    driver = Driver()
    assert isinstance(driver.name, str) and driver.name != ""
    assert driver.device_count >= 1


def test_driver_get_device() -> None:
    driver = Driver()
    device = driver.get_device(0)
    assert isinstance(device, Device)
    assert device.id == 0


def test_device_get_context() -> None:
    ctx = Driver().get_device(0).get_context()
    assert isinstance(ctx, Context)


def test_context_create_queue() -> None:
    ctx = Driver().get_device(0).get_context()
    queue = ctx.create_queue()
    assert isinstance(queue, Queue)
    queue.synchronize()


def test_context_create_stream() -> None:
    ctx = Driver().get_device(0).get_context()
    stream = ctx.create_stream()
    assert isinstance(stream, Stream)
    stream.synchronize()


def test_queue_record_event_and_synchronize() -> None:
    ctx = Driver().get_device(0).get_context()
    queue = ctx.create_queue()
    event = queue.record_event()
    assert isinstance(event, Event)
    event.synchronize()
    assert event.is_ready() is True


def test_stream_record_event_and_synchronize() -> None:
    ctx = Driver().get_device(0).get_context()
    stream = ctx.create_stream()
    event = stream.record_event()
    assert isinstance(event, Event)
    event.synchronize()
    assert event.is_ready() is True
