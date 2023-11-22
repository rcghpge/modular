# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#                                                                              #
# Runs an inference with a MO model, with custom ops                           #
#                                                                              #
# ===----------------------------------------------------------------------=== #
#
# Usage
# - Package custom op
#  $ mojo package custom_sqrt -o custom_sqrt.mojopkg
# - Run script
#  $ python mo-custom-ops.py --custom-ops-path=custom_sqrt.mojopkg
#    Without Custom Ops...
#    Output: [1. 1. 1. 1. 1.]
#    With Custom Ops...
#    Output: [42. 42. 42. 42. 42.]
#
# ===----------------------------------------------------------------------=== #

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from modular import engine


def without_custom_ops():
    print("Without Custom Ops...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = Path(script_dir, "mo-custom-op-model.mlir")
    session = engine.InferenceSession()
    model = session.load(model_path)
    output = model.execute(input=np.ones((5)))
    assert "output" in output.keys()
    print(f"Output: {output['output']}")


def with_custom_ops(custom_ops_path: str):
    print("With Custom Ops...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = Path(script_dir, "mo-custom-op-model.mlir")
    session = engine.InferenceSession()
    options = engine.CommonLoadOptions()
    options.custom_ops_path = custom_ops_path
    model = session.load(model_path, options)
    output = model.execute(input=np.ones((5)))
    assert "output" in output.keys()
    print(f"Output: {output['output']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--custom-ops-path",
        type=str,
        help="Path to mojo custom op package.",
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    without_custom_ops()

    args = parse_args()
    with_custom_ops(custom_ops_path=args.custom_ops_path)
