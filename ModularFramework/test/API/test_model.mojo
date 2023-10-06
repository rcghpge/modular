# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -I %engine_pkg_dir %s %S/mo.model | FileCheck %s

from aiengine import AIEngine
from sys import argv


fn test_model_compilation() raises:
    # CHECK: test_model_compilation
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

            let input_names = compiled_model.get_model_input_names()
            # CHECK: input
            for name in input_names:
                print(name)

            # CHECK: input
            print(input_names[0])

            # CHECK: 1
            print(compiled_model.num_model_outputs())

            let output_names = compiled_model.get_model_output_names()
            # CHECK: output
            for name in output_names:
                print(name)

            # CHECK: output
            print(output_names[0])


fn test_model() raises:
    # CHECK: test_model
    print("====test_model")

    let args = argv()

    # CHECK: 2
    print(len(args))

    # CHECK: mo.model
    print(args[1])

    let model_path = args[1]

    with AIEngine() as engine:
        with engine.get_session() as session:
            let model = session.load_model(model_path)


fn main() raises:
    test_model_compilation()
    test_model()
