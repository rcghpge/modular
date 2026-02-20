# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.pipelines.core import TTSContext
from max.pipelines.lib import (
    PIPELINE_REGISTRY,
    SupportedArchitecture,
)

from .config import LlasaConfig
from .pipeline_model import LlasaPipelineModel
from .tokenizer import LlasaTokenizer

llasa_arch = SupportedArchitecture(
    name="LlasaDecoder",
    task=PipelineTask.AUDIO_GENERATION,
    example_repo_ids=["HKUSTAudio/Llasa-8B"],
    default_encoding="bfloat16",
    supported_encodings={
        "bfloat16": ["paged"],
        "float32": ["paged"],
    },
    pipeline_model=LlasaPipelineModel,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=LlasaTokenizer,
    context_type=TTSContext,
    multi_gpu_supported=False,  # XCodec2 decoder currently uses single device
    config=LlasaConfig,
)


ARCHITECTURES = [llasa_arch]

# Auto-register the architecture when this module is imported
PIPELINE_REGISTRY.register(llasa_arch, allow_override=True)

__all__ = ["LlasaConfig", "llasa_arch"]
