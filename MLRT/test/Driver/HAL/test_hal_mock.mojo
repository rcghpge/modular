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
from std.sys._hal import Driver
from std.sys._hal.context import Context
from std.sys._hal.status import (
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
from std.testing import assert_equal, assert_true


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
        device_count = Int(driver.get_device_count())
    except e:
        raise Error(String.write(e))
    assert_equal(device_count, 2)


def test_driver_device_enumeration() raises:
    var ok = True
    try:
        var driver = Driver.create(_get_plugin_spec())
        for i in range(Int(driver.get_device_count())):
            var device = driver.get_device(Int64(i))
            if Int(device.id) != i:
                ok = False
    except e:
        raise Error(String.write(e))
    assert_true(ok, "device ID mismatch during enumeration")


def test_invalid_device_id() raises:
    var got_error = False
    try:
        var driver = Driver.create(_get_plugin_spec())
        _ = driver.get_device(Int64(99))
    except:
        got_error = True
    assert_true(got_error, "expected HALError for invalid device ID")


def test_context_create() raises:
    try:
        var driver = Driver.create(_get_plugin_spec())
        var device = driver.get_device(Int64(0))
        var ctx = Context(device)
        # Let ctx go out of scope to exercise Context destructor.
    except e:
        raise Error(String.write(e))


def test_invalid_device_error_message() raises:
    var msg = String("")
    try:
        var driver = Driver.create(_get_plugin_spec())
        _ = driver.get_device(Int64(99))
    except e:
        msg = String.write(e)
    assert_true(msg.byte_length() > 0, "expected non-empty error message")
    assert_true(
        "Invalid device ID" in msg,
        "error message should mention invalid device ID, got: " + msg,
    )


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
    test_driver_device_enumeration()
    test_invalid_device_id()
    test_context_create()
    test_invalid_device_error_message()
