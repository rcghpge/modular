# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -I %engine_pkg_dir %s | FileCheck %s

from aiengine import AIEngine, EngineDType


fn test_engine_version() raises:
    # CHECK: test_engine
    print("====test_engine")

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
    print(EngineDType(DType.int8) == EngineDType.int8)

    # CHECK: True
    print(EngineDType(DType.int8).to_dtype() == DType.int8)


fn main() raises:
    test_engine_version()
    test_status()
    test_session()
    test_dtype()
