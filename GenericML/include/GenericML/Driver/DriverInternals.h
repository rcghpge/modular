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
                      const TensorShape &userShape,
                      const QuantTensorSpec &modelSpec, Runtime &runtime);

/// Initialize raw pointers for each model input with the correct data fill.
/// Uses inputGenKind unless a per-tensor input source is specified in
/// inputDataSpecs. The input source can be one of:
/// 1. per-tensor InputGenerationKind
/// 2. per-tensor data from a file (stubbed out but not implemented yet)
/// The tensors can be accessed by calling getNextInput().
/// Returns ownership of tensor generators (containing tensors) to the caller.
ErrorOr<SmallVector<ITensorGeneratorPtr, 3>>
createTensorGenerators(InputGenerationKind inputGenKind,
                       ArrayRef<TensorShape> userInputShapes,
                       ArrayRef<InputDataSpec> inputDataSpecs,
                       const Model *model, Runtime &runtime);

void benchmarkModel(const Model *model, Runtime &runtime,
                    StringRef benchmarkName,
                    const ExecutorDriverCLOptions &options,
                    BenchmarkScenario scenario, StringRef outputDir,
                    bool benchmarkNoLogPrefix);
} // namespace M

#endif // DRIVERINTERNALS_H
