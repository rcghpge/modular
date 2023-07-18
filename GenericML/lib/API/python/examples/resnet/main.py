# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import numpy as np
from modular import engine
from pathlib import Path


def main():
    # Load the model
    print("Loading the model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = Path(script_dir, "resnet50_v1_savedmodel")
    session = engine.InferenceSession()
    model = session.load(model_path)

    # Run inference with random data
    input_tensor = np.random.rand(1, 224, 224, 3).astype(np.float32)
    print("Running inference...")
    output_tensor = model.execute(input=input_tensor)
    print(output_tensor)


if __name__ == "__main__":
    main()
