# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING

from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorType
from max.graph.weights import SafetensorWeights
from max.pipelines import PipelineConfig, TokenGenerator

from .vision_encoder.model import _build_graph


class PixtralVisionEncoder(TokenGenerator):
    """The Pixtral model."""

    def __init__(self, config: PipelineConfig, **kwargs):
        """Initializes the Llama3 model.

        Args:
            config: Model parameters.
            session: Optional InferenceSession to use to run the model.
            vocab_size: Vocabulary size of the model. Generally not required
              unless you're using an older model checkpoint.
        """
        self._config = config

        # Read in Hyperparameters.
        self._hf_config = self._config.huggingface_config

        # Load Device.
        self._device = self._config.device
        session = InferenceSession(devices=[self._device])

        # Load Weights from SafeTensors.
        if self._config.weight_path is None:
            raise ValueError(
                "no weight path provided for Pixtral based safetensors weights."
            )

        self._weights = SafetensorWeights([self._config.weight_path])  # type: ignore

        # Load Model.
        self._model = self._load_model(session)

    def _load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        if serialized_path := self._config.serialized_model_path:
            # Hydrate all weights to be referenced by the serialized graph.
            weights_registry = {}
            for name, tensor in self._weights._tensors.items():  # type: ignore
                weights_registry[name] = tensor.data
            print("Loading serialized model from", serialized_path, "...")
            return session.load(
                serialized_path,
                weights_registry=weights_registry,
            )
        else:
            print("Building model...")
            graph = _build_graph(self._config, self._hf_config, self._weights)
            print("Compiling...")
            return session.load(
                graph, weights_registry=self._weights.allocated_weights  # type: ignore
            )
