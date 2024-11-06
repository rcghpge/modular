# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Naive KV cache for the Transformer."""

from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import TensorType, BufferType, Graph
from .manager import KVCacheManager
from .cache_params import KVCacheParams


class NaiveKVCacheManager(KVCacheManager):
    def __init__(
        self,
        params: KVCacheParams,
        max_cache_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        device: Device,
        session: InferenceSession,
    ) -> None:
        super().__init__(
            params=params,
            max_cache_batch_size=max_cache_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            device=device,
            session=session,
        )

        self.keys = Tensor.zeros(
            shape=self.cache_shape,
            dtype=self.params.dtype,
            device=device,
        )

        self.values = Tensor.zeros(
            shape=self.cache_shape, dtype=self.params.dtype, device=device
        )

        self._increment_cache_length_graph = (
            self._create_increment_cache_lengths_graph(self.session)
        )

    @property
    def cache_shape(self) -> list[int]:
        return [
            self.max_seq_len,
            self.num_layers,
            self.max_cache_batch_size,
            self.params.n_kv_heads,
            self.params.head_dim,
        ]

    def _create_increment_cache_lengths_graph(self, session: InferenceSession):
        start_pos_type = TensorType(DType.int64, shape=[])
        tokens_type = TensorType(DType.int64, shape=["batch_size", "seq_len"])
        with Graph(
            "update_start_pos", input_types=[start_pos_type, tokens_type]
        ) as graph:
            start_pos, tokens = graph.inputs
            graph.output(start_pos + tokens.shape[1])

        return session.load(graph)

    def fetch(
        self, seq_ids: list[int]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        existing_keys = list(self.cache_lengths.keys())
        for i, seq_id in enumerate(seq_ids):
            if existing_keys[i] != seq_id:
                msg = (
                    "seq_ids passed, are different than current inflight"
                    " batch.Naive Caching currently does not support mutating"
                    " inflight batches."
                )
                raise ValueError(msg)

        return (
            self.keys,
            self.values,
            Tensor.scalar(self.max_sequence_length, DType.int64, self.device),
            # TODO: MSDK-1201 - This next variable is not used upstream.
            # It is included here, as a placeholder, until we can dynamically
            # return a number of tensors from both `fetch` and `input_symbols`.
            Tensor.scalar(self.max_sequence_length, DType.int64, self.device),
        )

    def input_symbols(
        self,
    ) -> tuple[TensorType, TensorType, TensorType, TensorType]:
        return (
            # k_cache
            BufferType(
                self.params.dtype,
                shape=[
                    self.max_seq_len,
                    self.num_layers,
                    "max_batch_size",
                    self.params.n_kv_heads,
                    self.params.head_dim,
                ],
            ),
            # v_cache
            BufferType(
                self.params.dtype,
                shape=[
                    self.max_seq_len,
                    self.num_layers,
                    "max_batch_size",
                    self.params.n_kv_heads,
                    self.params.head_dim,
                ],
            ),
            # start_pos
            TensorType(DType.int64, shape=[]),
            # null_op - this isnt used for the naive cache
            TensorType(DType.int64, shape=[]),
        )

    def increment_cache_lengths(
        self,
        kv_cache_inputs: tuple[TensorType, TensorType, TensorType, TensorType],
        prev_model_inputs: tuple[Tensor, ...],
    ) -> tuple[TensorType, TensorType, TensorType, TensorType]:
        """
        Prepare the inputs for a multistep execution, generally by incrementing
        the cache lengths. This should not require a device synchronization,
        as this would defeat the purpose of multistep execution.

        This should also not update the cache lengths in our manager, this batch is
        still considered in-progress.
        """
        k_cache, v_cache, start_pos, _ = kv_cache_inputs
        tokens, _ = prev_model_inputs

        new_start_pos = self._increment_cache_length_graph(start_pos, tokens)
        return (k_cache, v_cache, new_start_pos, new_start_pos)
