# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for the Mojo HAL bindings using a mock plugin.

Exercises the full HAL lifecycle (Driver -> Device -> Context) against a
lightweight C mock that implements the M_driver_* API without GPU hardware.
This catches FFI signature mismatches, status code drift, and basic error
handling paths.
"""

from std.os import getenv
from _hal import Driver
from _hal.context import Context
from _hal.queue import Queue
from _hal.stream import Stream
from _hal.event import Event, EVENT_FLAG_CPU_VISIBLE
from _hal.status import (
    STATUS_SUCCESS,
    STATUS_INCOMPATIBLE_VERSION,
    STATUS_UNKNOWN_ERROR,
    STATUS_ARGUMENT_OUT_OF_RANGE,
    STATUS_UNKNOWN_PROPERTY_NAME,
    STATUS_NO_DEVICES,
    STATUS_BAD_DRIVER,
    STATUS_BAD_DEVICE,
    STATUS_BAD_CONTEXT,
    STATUS_BAD_MEMORY,
    STATUS_BAD_QUEUE,
    STATUS_BAD_EVENT,
    STATUS_BAD_FUNCTION,
    STATUS_OUT_OF_MEMORY,
    STATUS_INVALID_ARG,
    STATUS_INCOMPATIBLE,
    STATUS_BAD_BUNDLE,
    STATUS_EXECUTE_MODE_NOT_SUPPORTED,
)
from std.testing import assert_equal, assert_raises, assert_true


def _get_plugin_spec() -> String:
    return getenv("MODULAR_DRIVER_PLUGINS")


# ===-----------------------------------------------------------------------===#
# Status code value tests
# ===-----------------------------------------------------------------------===#


def test_status_code_values() raises:
    assert_equal(Int(STATUS_SUCCESS), 0)
    assert_equal(Int(STATUS_INCOMPATIBLE_VERSION), -1)
    assert_equal(Int(STATUS_UNKNOWN_ERROR), -2)
    assert_equal(Int(STATUS_ARGUMENT_OUT_OF_RANGE), -3)
    assert_equal(Int(STATUS_UNKNOWN_PROPERTY_NAME), -4)
    assert_equal(Int(STATUS_NO_DEVICES), -5)
    assert_equal(Int(STATUS_BAD_DRIVER), -6)
    assert_equal(Int(STATUS_BAD_DEVICE), -7)
    assert_equal(Int(STATUS_BAD_CONTEXT), -8)
    assert_equal(Int(STATUS_BAD_MEMORY), -9)
    assert_equal(Int(STATUS_BAD_QUEUE), -10)
    assert_equal(Int(STATUS_BAD_EVENT), -11)
    assert_equal(Int(STATUS_BAD_FUNCTION), -12)
    assert_equal(Int(STATUS_OUT_OF_MEMORY), -13)
    assert_equal(Int(STATUS_INVALID_ARG), -14)
    assert_equal(Int(STATUS_INCOMPATIBLE), -15)
    assert_equal(Int(STATUS_BAD_BUNDLE), -16)
    assert_equal(Int(STATUS_EXECUTE_MODE_NOT_SUPPORTED), -17)


# ===-----------------------------------------------------------------------===#
# Driver lifecycle tests
# ===-----------------------------------------------------------------------===#


def test_driver_create() raises:
    var device_count: Int
    try:
        var driver = Driver.create(_get_plugin_spec())
        device_count = Int(driver[].get_device_count())
    except e:
        raise Error(String.write(e))
    assert_equal(device_count, 2)


def test_invalid_device_id() raises:
    var driver = Driver.create(_get_plugin_spec())
    with assert_raises():
        _ = driver[].get_device[Int64(99)]()


def test_context_create() raises:
    try:
        var driver = Driver.create(_get_plugin_spec())
        var device = driver[].get_device[Int64(0)]()
        var ctx = device[].get_context()
        # Let ctx go out of scope to exercise Context destructor.
    except e:
        raise Error(String.write(e))


def test_invalid_device_error_message() raises:
    var driver = Driver.create(_get_plugin_spec())
    with assert_raises(contains="Invalid device ID"):
        _ = driver[].get_device[Int64(99)]()


# ===-----------------------------------------------------------------------===#
# Queue tests
# ===-----------------------------------------------------------------------===#


def test_queue_create_and_sync() raises:
    try:
        var driver = Driver.create(_get_plugin_spec())
        var device = driver[].get_device[Int64(0)]()
        var ctx = device[].get_context()
        var queue = ctx[].create_queue()
        queue[].synchronize()
    except e:
        raise Error(String.write(e))


def test_queue_record_event() raises:
    try:
        var driver = Driver.create(_get_plugin_spec())
        var device = driver[].get_device[Int64(0)]()
        var ctx = device[].get_context()
        var queue = ctx[].create_queue()
        var event = queue[].record_event()
        queue[].wait_for_events(event)
        queue[].synchronize()
    except e:
        raise Error(String.write(e))


def test_queue_wait_for_events() raises:
    try:
        var driver = Driver.create(_get_plugin_spec())
        var device = driver[].get_device[Int64(0)]()
        var ctx = device[].get_context()
        var queue1 = ctx[].create_queue()
        var queue2 = ctx[].create_queue()
        var e1 = queue1[].record_event()
        var e2 = queue1[].record_event()
        queue2[].wait_for_events(e1, e2)
        queue2[].synchronize()
    except e:
        raise Error(String.write(e))


def test_queue_wait_for_events_mixed_flags() raises:
    """Variadic accepts events with different flag combos in one call."""
    try:
        var driver = Driver.create(_get_plugin_spec())
        var device = driver[].get_device[Int64(0)]()
        var ctx = device[].get_context()
        var queue1 = ctx[].create_queue()
        var queue2 = ctx[].create_queue()
        var e_intra = queue1[].record_event()
        var e_host = queue1[].record_event[EVENT_FLAG_CPU_VISIBLE]()
        queue2[].wait_for_events(e_intra, e_host)
        queue2[].synchronize()
    except e:
        raise Error(String.write(e))


# ===-----------------------------------------------------------------------===#
# Stream tests
# ===-----------------------------------------------------------------------===#


def test_stream_create_and_sync() raises:
    try:
        var driver = Driver.create(_get_plugin_spec())
        var device = driver[].get_device[Int64(0)]()
        var ctx = device[].get_context()
        var stream = ctx[].create_stream()
        stream[].synchronize()
    except e:
        raise Error(String.write(e))


def test_stream_record_event() raises:
    try:
        var driver = Driver.create(_get_plugin_spec())
        var device = driver[].get_device[Int64(0)]()
        var ctx = device[].get_context()
        var stream = ctx[].create_stream()
        var event = stream[].record_event()
        stream[].wait_for_events(event)
        stream[].synchronize()
    except e:
        raise Error(String.write(e))


def test_stream_wait_for_events() raises:
    try:
        var driver = Driver.create(_get_plugin_spec())
        var device = driver[].get_device[Int64(0)]()
        var ctx = device[].get_context()
        var producer = ctx[].create_stream()
        var consumer = ctx[].create_stream()
        var ev = producer[].record_event()
        consumer[].wait_for_events(ev)
        consumer[].synchronize()
    except e:
        raise Error(String.write(e))


# ===-----------------------------------------------------------------------===#
# Entry point
# ===-----------------------------------------------------------------------===#


def main() raises:
    test_status_code_values()

    var spec = _get_plugin_spec()
    if not spec:
        print("SKIP: MODULAR_DRIVER_PLUGINS not set")
        return

    test_driver_create()
    test_invalid_device_id()
    test_context_create()
    test_invalid_device_error_message()
    test_queue_create_and_sync()
    test_queue_record_event()
    test_queue_wait_for_events()
    test_queue_wait_for_events_mixed_flags()
    test_stream_create_and_sync()
    test_stream_record_event()
    test_stream_wait_for_events()
