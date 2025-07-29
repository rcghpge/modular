# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import msgspec
import numpy as np
from max.interfaces import msgpack_numpy_decoder, msgpack_numpy_encoder
from pytest_benchmark.fixture import BenchmarkFixture


class SampleData(msgspec.Struct, omit_defaults=True):
    data: np.ndarray = msgspec.field()


# The number of float32 values in 64mb
FLOAT_32_64MB = 16257408
FLOAT_32_32MB = int(FLOAT_32_64MB / 2)
FLOAT_32_24MB = int(FLOAT_32_64MB * 3 / 8)
FLOAT_32_16MB = int(FLOAT_32_64MB / 4)
FLOAT_32_8MB = int(FLOAT_32_64MB / 8)
SHARED_MEMORY_ENCODER = msgpack_numpy_encoder(
    use_shared_memory=True, shared_memory_threshold=0
)
NON_SHARED_MEMORY_ENCODER = msgpack_numpy_encoder(use_shared_memory=False)
DECODER = msgpack_numpy_decoder(SampleData)


def test_non_shared_memory_serialization_64MB(
    benchmark: BenchmarkFixture,
) -> None:
    sample_data = SampleData(np.random.rand(FLOAT_32_64MB).astype(np.float32))

    def test_encode_and_decode() -> None:
        encoded = NON_SHARED_MEMORY_ENCODER(sample_data)
        decoded = DECODER(encoded)

    benchmark(test_encode_and_decode)


def test_shared_memory_serialization_64MB(
    benchmark: BenchmarkFixture,
) -> None:
    sample_data = SampleData(np.random.rand(FLOAT_32_64MB).astype(np.float32))

    def test_encode_and_decode() -> None:
        encoded = SHARED_MEMORY_ENCODER(sample_data)
        decoded = DECODER(encoded)

    benchmark(test_encode_and_decode)


def test_non_shared_memory_serialization_32MB(
    benchmark: BenchmarkFixture,
) -> None:
    sample_data = SampleData(np.random.rand(FLOAT_32_32MB).astype(np.float32))

    def test_encode_and_decode() -> None:
        encoded = NON_SHARED_MEMORY_ENCODER(sample_data)
        decoded = DECODER(encoded)

    benchmark(test_encode_and_decode)


def test_shared_memory_serialization_32MB(
    benchmark: BenchmarkFixture,
) -> None:
    sample_data = SampleData(np.random.rand(FLOAT_32_32MB).astype(np.float32))

    def test_encode_and_decode() -> None:
        encoded = SHARED_MEMORY_ENCODER(sample_data)
        decoded = DECODER(encoded)

    benchmark(test_encode_and_decode)


def test_non_shared_memory_serialization_24MB(
    benchmark: BenchmarkFixture,
) -> None:
    sample_data = SampleData(np.random.rand(FLOAT_32_24MB).astype(np.float32))

    def test_encode_and_decode() -> None:
        encoded = NON_SHARED_MEMORY_ENCODER(sample_data)
        decoded = DECODER(encoded)

    benchmark(test_encode_and_decode)


def test_shared_memory_serialization_24MB(
    benchmark: BenchmarkFixture,
) -> None:
    sample_data = SampleData(np.random.rand(FLOAT_32_24MB).astype(np.float32))

    def test_encode_and_decode() -> None:
        encoded = SHARED_MEMORY_ENCODER(sample_data)
        decoded = DECODER(encoded)

    benchmark(test_encode_and_decode)


def test_non_shared_memory_serialization_16MB(
    benchmark: BenchmarkFixture,
) -> None:
    sample_data = SampleData(np.random.rand(FLOAT_32_16MB).astype(np.float32))

    def test_encode_and_decode() -> None:
        encoded = NON_SHARED_MEMORY_ENCODER(sample_data)
        decoded = DECODER(encoded)

    benchmark(test_encode_and_decode)


def test_shared_memory_serialization_16MB(
    benchmark: BenchmarkFixture,
) -> None:
    sample_data = SampleData(np.random.rand(FLOAT_32_16MB).astype(np.float32))

    def test_encode_and_decode() -> None:
        encoded = SHARED_MEMORY_ENCODER(sample_data)
        decoded = DECODER(encoded)

    benchmark(test_encode_and_decode)


def test_non_shared_memory_serialization_8MB(
    benchmark: BenchmarkFixture,
) -> None:
    sample_data = SampleData(np.random.rand(FLOAT_32_8MB).astype(np.float32))

    def test_encode_and_decode() -> None:
        encoded = NON_SHARED_MEMORY_ENCODER(sample_data)
        decoded = DECODER(encoded)

    benchmark(test_encode_and_decode)


def test_shared_memory_serialization_8MB(
    benchmark: BenchmarkFixture,
) -> None:
    sample_data = SampleData(np.random.rand(FLOAT_32_8MB).astype(np.float32))

    def test_encode_and_decode() -> None:
        encoded = SHARED_MEMORY_ENCODER(sample_data)
        decoded = DECODER(encoded)

    benchmark(test_encode_and_decode)
