# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Runs an inference with a BERT model, using dummy data, just as a demo.

import os
from pathlib import Path

import numpy as np
from max import engine


def main():
    print("Loading the model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = Path(script_dir, "bert-base-uncased")
    session = engine.InferenceSession()
    model = session.load(model_path)

    print("Running inference...")
    output = model.execute(
        np.zeros((1, 1), dtype=np.int32),
        np.zeros((1, 1), dtype=np.int32),
        np.zeros((1, 1), dtype=np.int32),
    )
    print([o.to_numpy() for o in output])


if __name__ == "__main__":
    main()
