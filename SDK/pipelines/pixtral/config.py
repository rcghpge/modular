# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""All configurable parameters for Pixtral."""

from max.pipelines import HuggingFaceFile, SupportedEncoding


def get_pixtral_huggingface_file(
    encoding: SupportedEncoding,
) -> HuggingFaceFile:
    if encoding == SupportedEncoding.float32:
        raise ValueError(f"Pixtral does not support: {encoding}")
    elif encoding == SupportedEncoding.bfloat16:
        return HuggingFaceFile(
            "mistralai/Pixtral-12B-2409",
            "consolidated.safetensors",
        )
    else:
        raise ValueError(f"Pixtral does not support: {encoding}")
