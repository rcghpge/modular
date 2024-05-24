//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines helper functions used within the ExecutionDriver library.
//
//===----------------------------------------------------------------------===//

#ifndef DRIVERINTERNALS_H
#define DRIVERINTERNALS_H

#include "GenericML/Driver/DriverCLOptions.h"
#include "GenericML/Support/Model.h"
#include "Support/ForwardDecls.h"

namespace M {

class ITensorGenerator {
public:
  virtual const Tensor &getNextTensor() = 0;
  virtual const Tensor &getTensor(size_t sampleIdx) const = 0;
  virtual size_t getSampleCount() const = 0;
  virtual bool isCycled() const = 0;
  virtual ~ITensorGenerator() = default;
};

using ITensorGeneratorPtr = std::unique_ptr<ITensorGenerator>;

ErrorOr<ITensorGeneratorPtr>
createTensorGenerator(size_t tensorIdx, InputGenerationKind defaultGen,
                      ArrayRef<InputDataSpec> inputDataSpecs,
                      const QuantTensorSpec &modelSpec,
                      size_t minimumCpuAlignment, Runtime &runtime);

/// Initialize raw pointers for each model input with the correct data fill.
/// Uses inputGenKind unless a per-tensor input source is specified in
/// inputDataSpecs. The input source can be one of:
/// 1. per-tensor InputGenerationKind
/// 2. per-tensor data from a file (stubbed out but not implemented yet)
/// The tensors can be accessed by calling getNextInput().
/// Returns ownership of tensor generators (containing tensors) to the caller.
ErrorOr<SmallVector<ITensorGeneratorPtr, 3>>
createTensorGenerators(InputGenerationKind inputGenKind,
                       ArrayRef<QuantTensorSpec> inputShapeSpecs,
                       ArrayRef<InputDataSpec> inputDataSpecs,
                       size_t minimumCpuAlignment, Runtime &runtime);

ErrorOrSuccess
benchmarkModel(std::unique_ptr<ExecutableModel> model, Runtime &runtime,
               StringRef benchmarkName, const ExecutorDriverOptions &options,
               ArrayRef<QuantTensorSpec> inputSpecs, BenchmarkScenario scenario,
               StringRef outputDir, bool benchmarkNoLogPrefix,
               uint64_t min_duration_ms, uint64_t min_query_count);
} // namespace M

#endif // DRIVERINTERNALS_H
