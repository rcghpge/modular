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

from typing import Any

import msgspec
import numpy as np
import numpy.typing as npt
from max.pipelines.core.context import TextAndVisionContext


class VisionEncodingData(
    msgspec.Struct, tag=True, kw_only=True, omit_defaults=True
):
    """Container for vision-specific encoding data used during image/video processing.

    This data is only present when the context includes images or videos that need
    to be encoded by the vision tower.
    """

    # Grid and temporal parameters
    image_grid_thw: npt.NDArray[np.int32]
    video_grid_thw: npt.NDArray[np.int32] | None
    second_per_grid_ts: npt.NDArray[np.float32] | None

    # Vision-specific position and window parameters
    vision_position_ids: npt.NDArray[np.integer[Any]]
    window_index: npt.NDArray[np.integer[Any]]
    max_grid_size: npt.NDArray[np.int32]

    # Sequence length parameters for attention
    cu_seqlens: npt.NDArray[np.uint32]
    cu_window_seqlens_unique: npt.NDArray[np.uint32]
    max_seqlen: npt.NDArray[np.uint32]
    window_max_seqlen: npt.NDArray[np.uint32]

    # Vision inputs (pixel values for images/videos)
    concatenated_pixel_values: npt.NDArray[np.float32]


class Qwen2_5VLTextAndVisionContext(
    TextAndVisionContext, tag=True, kw_only=True, omit_defaults=True
):
    """Context object for Qwen2.5VL multimodal model processing.

    Extends TextAndVisionContext with Qwen2.5VL-specific configuration and state.
    Vision encoding data is stored in an optional VisionEncodingData object that
    is only present when images/videos need to be encoded.
    """

    # Token and configuration parameters (scalar values)
    spatial_merge_size: int
    rope_delta: int
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    tokens_per_second: int

    # Token indices and position IDs (arrays)
    image_token_indices: npt.NDArray[np.int32]
    decoder_position_ids: npt.NDArray[np.int64]

    # Vision encoding data (only present when encoding images/videos)
    vision_data: VisionEncodingData | None
