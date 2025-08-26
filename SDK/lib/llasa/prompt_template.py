# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
from transformers import AutoTokenizer

from .config import LlasaConfig


def create_llasa_prompt(
    text: str, config: LlasaConfig
) -> tuple[np.ndarray, AutoTokenizer]:
    """Create Llasa-formatted prompt from text. The bulk of this code is copied
    directly from the HF model card, with some changes made to avoid returning
    PyTorch tensors."""
    tokenizer = AutoTokenizer.from_pretrained(config.llama_model_path)

    formatted_text = f"{config.text_understanding_start}{text}{config.text_understanding_end}"

    chat = [
        {
            "role": "user",
            "content": "Convert the text to speech:" + formatted_text,
        },
        {"role": "assistant", "content": config.speech_generation_start},
    ]

    input_ids = tokenizer.apply_chat_template(
        chat, tokenize=True, return_tensors=None, continue_final_message=True
    )
    input_ids_array = np.array(input_ids, dtype=np.int64)

    return input_ids_array, tokenizer
