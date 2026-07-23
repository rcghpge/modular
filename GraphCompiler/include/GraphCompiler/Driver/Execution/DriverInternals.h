//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines helper functions used within the ExecutionDriver library.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_EXECUTIONDRIVER_DRIVERINTERNALS_H
#define GENERICML_EXECUTIONDRIVER_DRIVERINTERNALS_H

#include "GraphCompiler/Driver/Execution/DriverCLOptions.h"
#include "GraphCompiler/Support/QuantSpec.h"
#include "Support/ForwardDecls.h"

namespace M {

class ITensorGenerator {
public:
  virtual const MLRT::Tensor &getNextTensor() = 0;
  virtual const MLRT::Tensor &getTensor(size_t sampleIdx) const = 0;
  virtual size_t getSampleCount() const = 0;
  virtual bool isCycled() const = 0;
  virtual ~ITensorGenerator() = default;
};

using ITensorGeneratorPtr = std::unique_ptr<ITensorGenerator>;

ErrorOr<ITensorGeneratorPtr>
createTensorGenerator(size_t tensorIdx, InputGenerationKind defaultGen,
                      ArrayRef<InputDataSpec> inputDataSpecs,
                      const QuantTensorSpec &modelSpec,
                      size_t minimumCpuAlignment, CPUDevice &cpuDevice);

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
                       size_t minimumCpuAlignment, CPUDevice &cpuDevice);

} // namespace M

#endif // GENERICML_EXECUTIONDRIVER_DRIVERINTERNALS_H
