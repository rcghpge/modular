# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import logging
import time

from max.driver import Device, Tensor
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.graph.weights import Weights, WeightsAdapter
from max.interfaces import InputContext
from max.nn import ReturnLogits
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
)
from transformers import AutoConfig

from .graph import build_graph

logger = logging.getLogger("max.pipelines")


class WhisperInputs(ModelInputs):
    """A class representing inputs for the Whisper model.

    input_features:
        Float values mel features extracted from the raw speech waveform.
        Raw speech waveform can be obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* viathe soundfile library (`pip install soundfile`).
        To prepare the array into `input_features`, the [`AutoFeatureExtractor`] from the transformers library should be used for extracting the mel features, padding and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        Shape = (batch_size, feature_size, sequence_length)

    decoder_input_ids:
        Indices of decoder input sequence tokens in the vocabulary. Indices can be obtained using [`WhisperTokenizer`].
        Whisper uses the `decoder_start_token_id` as the starting token for `decoder_input_ids` generation.
        Shape = (batch_size, target_sequence_length)
    """

    input_features: Tensor
    decoder_input_ids: Tensor


# TODO: Need specific InputContext type, not just this base type.
class Whisper(PipelineModel[InputContext]):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )
        self.model = self.load_model(session)

    def load_model(self, session: InferenceSession) -> Model:
        """
        Load the Whisper speech recognition model.
        """
        logger.info("Building and compiling Whisper encoder-decoder model...")
        before = time.perf_counter()
        if self.adapter:
            state_dict = self.adapter(dict(self.weights.items()))
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }
        graph = build_graph(
            state_dict,
            self.huggingface_config,
            self.encoding.dtype,
            DeviceRef.from_device(self.devices[0]),
        )
        after_build = time.perf_counter()

        logger.info(f"Building graph took {after_build - before:.6f} seconds")

        before_compile = time.perf_counter()
        model = session.load(graph, weights_registry=state_dict)
        after = time.perf_counter()

        logger.info(
            f"Compiling model took {after - before_compile:.6f} seconds"
        )

        logger.info(
            f"Building and compiling Whisper model took {after - before:.6f} seconds"
        )

        return model
