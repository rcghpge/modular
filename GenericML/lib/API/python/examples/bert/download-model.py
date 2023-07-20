# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Downloads a model (HF_MODEL_NAME) from Hugging Face and converts it to
# TensorFlow SavedModel format suitable for Modular Inference Engine.

import os
from transformers import TFAutoModel
from pathlib import Path

HF_MODEL_NAME = "bert-base-uncased"


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = Path(script_dir, "bert-base-uncased")
    if os.path.exists(model_path):
        print(HF_MODEL_NAME, "already exists.")
        return

    print("Converting model:", HF_MODEL_NAME)
    model = TFAutoModel.from_pretrained(HF_MODEL_NAME)
    model.save(model_path)


if __name__ == "__main__":
    main()
