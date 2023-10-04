# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -I %mof_pkg_dir %s %S/mo.model | FileCheck %s

from mof import AIEngine
from sys import argv


fn test_model_compilation() raises:
    print("====test_model_compilation")

    let args = argv()

    # CHECK: 2
    print(len(args))

    # CHECK: mo.model
    print(args[1])

    let model_path = args[1]

    with AIEngine() as engine:
        with engine.get_session() as session:
            let compiled_model = session.compile_model(model_path)
            # CHECK: 1
            print(compiled_model.num_model_inputs())


fn main() raises:
    test_model_compilation()
