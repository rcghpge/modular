# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Runs an inference with a BERT model, using dummy data, just as a demo.

import os
import numpy as np
from modular import engine
from pathlib import Path


def main():
    print("Loading the model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = Path(script_dir, "bert-base-uncased")
    session = engine.InferenceSession()
    model = session.load(model_path)

    print("Running inference...")
    output = model.execute(
        attention_mask=np.zeros((1, 1), dtype=np.int32),
        input_ids=np.zeros((1, 1), dtype=np.int32),
        token_type_ids=np.zeros((1, 1), dtype=np.int32),
    )
    print(output)


if __name__ == "__main__":
    main()
