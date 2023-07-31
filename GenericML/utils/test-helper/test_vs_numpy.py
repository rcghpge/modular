# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import argparse
import numpy as np
from test_cases import get_test_case
from ops import NumpyImpl
from tempfile import NamedTemporaryFile
import subprocess
import re


"""
This tool executes a MO graph using `mt` and compares it to the numpy
reference implementation.
"""


parser = argparse.ArgumentParser(
    prog="test_vs_numpy",
    description="From the given python test case create a MO graph",
)
parser.add_argument(
    "-n", "--name", required=True, help="Name of the test case to export"
)
parser.add_argument(
    "-f", "--file-to-run", required=True, help="MGP file to execute"
)
args = parser.parse_args()


def test_vs_numpy(testCase, file_to_run):
    np.random.seed(42)

    golden_value = []

    inputs = None
    num_inputs = None

    sizes_to_test = testCase.input_sizes_to_test()
    assert len(sizes_to_test) == 1, "Only one input config supported right now."

    # Create the input data + golden test values.
    for i in range(testCase.num_runs()):
        ins = testCase.generate_inputs(sizes_to_test[0])
        # Create list lazily so we know how many to create.
        if inputs is None:
            num_inputs = len(ins)
            inputs = [[] for _ in range(num_inputs)]

        for i in range(num_inputs):
            inputs[i].append(ins[i])
        golden_value.append(testCase.run(ins, NumpyImpl()))

    if not num_inputs or not inputs:
        print("No inputs to test!")
        exit(0)

    # Concat all the test data into one numpy array for each input and save out to a file which mt can load.
    inputs_as_numpy = [np.array(inputs[i]) for i in range(num_inputs)]

    input_launch_line = ""

    tmp_files = []
    for i in range(num_inputs):
        tmp = NamedTemporaryFile(suffix=".bin")
        tmp_files.append(tmp)
        inputs_as_numpy[i].tofile(tmp)

        # Save to file and flush so MT sees it.
        tmp.flush()

        shape = inputs_as_numpy[i].shape
        shape = "x".join([str(s) for s in shape[1:]])

        # Add the input to the MT command.
        input_launch_line += (
            f" --input-data={i}::{tmp.name} --input-shapes={shape}"
        )

    # Execute with mt to get the MO output.
    out = subprocess.run(
        (
            f"mt --execute {file_to_run} --num-runs={testCase.num_runs()}"
            f" {input_launch_line} --result-output-style=full"
        ),
        capture_output=True,
        shell=True,
    )

    if out.returncode != 0:
        print("mt failed to execute given file")
        print("stdout:")
        print(out.stdout)
        print("stderr:")
        print(out.stderr)
        assert False, "MT failed to run"

    lines = out.stdout.splitlines()

    output_index = 0
    # Iterate over all outputs.
    for line in out.stdout.splitlines()[1:]:
        if len(line) == 0:
            continue

        line_as_str = str(line)

        x = re.findall(r"tensor<(\d+(?:x\d+)*)", line_as_str)

        # Check we returned the same number of tensors as the golden value returned.
        assert len(x) == 1

        val_to_compare = golden_value[output_index]

        # Check the dimensions match.
        for tensor in x:
            mo_shape = tensor.split("x")
            for mo_dim, np_dim in zip(mo_shape, val_to_compare.shape):
                if int(mo_dim) != np_dim:

                    print(val_to_compare)
                    print(line_as_str)
                    print("Inputs: ")
                    for inp in inputs_as_numpy:
                        print(inp[output_index])

                    print(
                        f"On output {output_index} dimensions don't match! Mo"
                        f" {mo_shape} vs np {val_to_compare.shape}"
                    )
                    exit(0)

        # The raw output data returned by the test
        raw_out = line_as_str[
            line_as_str.find("[") + 1 : line_as_str.find("]")
        ].split(", ")

        mo_out = np.array(list(raw_out), dtype=val_to_compare.dtype).reshape(
            val_to_compare.shape
        )

        if not testCase.compare_results(mo_out, val_to_compare):
            print(f"Output at {output_index} does not match!")
            exit(0)

        output_index += 1

    print("All MO values match numpy")


test_vs_numpy(get_test_case(args.name), args.file_to_run)
