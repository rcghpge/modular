# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Basic smoke test for the Mojo HAL driver.

Requires MODULAR_DRIVER_PLUGINS env var set to a plugin spec, e.g.:
    CUDA@/path/to/cuda_plugin.so
"""

from std.os import getenv
from _hal import Driver
from _hal.status import HALError, STATUS_UNKNOWN_ERROR


def test_driver_create() raises HALError:
    var plugin_spec = getenv("MODULAR_DRIVER_PLUGINS")
    if not plugin_spec:
        print("SKIP: MODULAR_DRIVER_PLUGINS not set")
        return

    var driver = Driver.create(plugin_spec)
    print("Driver name:", driver[].get_name())

    var device_count = driver[].get_device_count()
    print("Device count:", device_count)


def test_device_get() raises HALError:
    var plugin_spec = getenv("MODULAR_DRIVER_PLUGINS")
    if not plugin_spec:
        print("SKIP: MODULAR_DRIVER_PLUGINS not set")
        return

    var driver = Driver.create(plugin_spec)
    var device_count = driver[].get_device_count()
    for i in range(device_count):
        var device = driver[].get_device_dynamic(Int64(i))
        if device[].id != Int64(i):
            raise HALError(
                STATUS_UNKNOWN_ERROR,
                message="Device ID mismatch: expected "
                + String(i)
                + ", got "
                + String(device[].id),
            )
        print("Device", i, "OK")


def main() raises:
    try:
        test_driver_create()
    except e:
        raise Error(String.write(e))
