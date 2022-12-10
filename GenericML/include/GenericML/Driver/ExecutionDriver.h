//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides an interface to the standardized "frwk-execute" command
// line tools.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_EXECUTIONDRIVER_H
#define GENERICML_DRIVER_EXECUTIONDRIVER_H

#include "LLCL/ForwardDecls.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include <cstddef>

namespace M {
class Model;
class Tensor;
class ExecutorDriverCLOptions;

enum class InputGenerationKind {
  kRandom, // generate inputs with random elements, distribution depends on type
  kZeros,  // generate inputs filled with zeros
  kOnes,   // generate inputs filled with ones
  kNegOnes, // generate inputs filled with negative ones
};

enum class BenchmarkScenario { kOnline, kBatched };

/// Run `model` using the command line options flags configured by the user.
LogicalResult runModel(MLIRContext *ctx, const Model *model,
                       LLCL::Runtime &runtime,
                       const ExecutorDriverCLOptions &options);

/// Run `model` using the command line options flags configured by the user.
/// `inputs` is the input tensor data to the model, while `outputs` is
/// pre-allocated memory to be filled with the model's results.
/// TODO(#5867): This API could do with some love.
ErrorOrSuccess runModelWithData(MLIRContext *ctx, const Model *model,
                                LLCL::Runtime &runtime,
                                const ExecutorDriverCLOptions &options,
                                const llvm::SmallVector<const void *> &inputs,
                                llvm::SmallVector<void *> &outputs);

} // namespace M

#endif // GENERICML_DRIVER_EXECUTIONDRIVER_H
