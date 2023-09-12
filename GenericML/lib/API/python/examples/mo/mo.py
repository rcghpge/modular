# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Runs an inference with a MO model, using dummy data

import os
import numpy as np
from modular import engine
from pathlib import Path


def main():
    print("Loading MO model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = Path(script_dir, "mo.mlir")
    session = engine.InferenceSession()
    model = session.load(model_path)

    print("Running inference...")
    output = model.execute(input=np.random.rand(5).astype(np.float32))
    assert "output" in output.keys()
    print("Output dtype: ", output["output"].dtype)
    print("Output shape: ", output["output"].shape)


if __name__ == "__main__":
    main()
