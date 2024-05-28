//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SDK/EngineCLI/Benchmark/Tensor.h"
#include "Support/AssertStream.h"

using namespace M;
using namespace M::BenchmarkCLI;

M::BenchmarkCLI::Tensor::Tensor() {}
M::BenchmarkCLI::Tensor::~Tensor() {}

M::BenchmarkCLI::Tensor::Tensor(Tensor &&) = default;
Tensor &M::BenchmarkCLI::Tensor::operator=(Tensor &&) = default;

M::BenchmarkCLI::Tensor::Tensor(TensorSpec spec, const void *data,
                                std::unique_ptr<DataOwner> dataOwner)
    : spec(std::move(spec)), data(data), dataOwner(std::move(dataOwner)) {
  ASSERT_STREAM(spec.isStatic())
      << "Tensor must be constructed with static shape";
}
