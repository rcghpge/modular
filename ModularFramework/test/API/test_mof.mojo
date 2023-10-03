# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo -I %mof_pkg_dir %s | FileCheck %s

from mof import AIEngine


fn test_mof_version() raises:
    # CHECK: test_mof
    print("====test_mof")

    let engine = AIEngine()

    # CHECK: Version: {{.*}}
    print("Version:", engine.get_version())


fn test_status() raises:
    # CHECK: test_status
    print("====test_status")

    let engine = AIEngine()
    let status = engine.get_status()

    # CHECK: No Error!
    if status:
        print(status.__str__())
    else:
        print("No Error!")

    _ = engine ^


fn main() raises:
    test_mof_version()
    test_status()
