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
#include "llvm/ADT/SmallVector.h"
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

StringLiteral stringifyEnum(InputGenerationKind kind);
std::optional<InputGenerationKind> symbolizeInputGenerationKind(StringRef s);

enum class TensorGeneratorFlag : int {
  kCycle = 1 << 0,    // generate infinite number of Tensors from generator by
                      // looping over file repeatedly
  kTruncate = 1 << 1, // if input file size not a multiple of tensor
                      // size, discard the remainder
};

inline TensorGeneratorFlag operator|(TensorGeneratorFlag a,
                                     TensorGeneratorFlag b) {
  return static_cast<TensorGeneratorFlag>(static_cast<int>(a) |
                                          static_cast<int>(b));
}

inline bool isSet(TensorGeneratorFlag a, TensorGeneratorFlag b) {
  return static_cast<int>(a) & static_cast<int>(b);
}

std::optional<TensorGeneratorFlag> symbolizeTensorGeneratorFlag(StringRef s);

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
