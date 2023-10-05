# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -I %mof_pkg_dir %s | FileCheck %s

from mof import AIEngine, MOFDtype


fn test_mof_version() raises:
    # CHECK: test_mof
    print("====test_mof")

    let engine = AIEngine()

    # CHECK: Version: {{.*}}
    print("Version:", engine.get_version())


fn test_status() raises:
    # CHECK: test_status
    print("====test_status")

    with AIEngine() as engine:
        let status = engine.get_status()

        # CHECK: No Error!
        if status:
            print(status.__str__())
        else:
            print("No Error!")


fn test_session() raises:
    # CHECK: test_session
    print("====test_session")

    with AIEngine() as engine:
        let session = engine.get_session()


fn test_dtype() raises:
    # CHECK: test_dtype
    print("====test_dtype")

    # CHECK: True
    print(MOFDtype(DType.int8) == MOFDtype.int8)

    # CHECK: True
    print(MOFDtype(DType.int8).to_dtype() == DType.int8)


fn main() raises:
    test_mof_version()
    test_status()
    test_session()
    test_dtype()
