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
from _hal.extensions.traits import SupportsPeerAccessV1
from _hal.status import HALError, STATUS_UNKNOWN_ERROR
from std.testing import assert_equal, assert_false


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


def test_peer_access() raises:
    var plugin_spec = getenv("MODULAR_DRIVER_PLUGINS")
    if not plugin_spec:
        print("SKIP: MODULAR_DRIVER_PLUGINS not set")
        return

    var driver = Driver.create(plugin_spec)
    if driver[].get_device_count() < 2:
        print("single device; skipping")
        return
    var dev0 = driver[].get_device_dynamic(Int64(0))
    var dev1 = driver[].get_device_dynamic(Int64(1))
    comptime Ext = type_of(dev0[].extensions)
    comptime if conforms_to(Ext, SupportsPeerAccessV1):
        assert_false(dev0[].can_access_peer(dev0[]))
        var forward = dev0[].can_access_peer(dev1[])
        assert_equal(forward, dev1[].can_access_peer(dev0[]))
        if not forward:
            print("pair has no peer access; skipping enable")
            return
        var ctx0 = dev0[].get_context()
        var ctx1 = dev1[].get_context()
        ctx0[].enable_peer_access(ctx1[])
        ctx0[].enable_peer_access(ctx1[])
    else:
        print("extension set has no peer access; skipping")


def main() raises:
    try:
        test_driver_create()
    except e:
        raise Error(String(e))
    test_peer_access()
