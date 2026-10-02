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

#ifndef GENERICML_EXECUTIONDRIVER_EXECUTIONDRIVER_H
#define GENERICML_EXECUTIONDRIVER_EXECUTIONDRIVER_H

#include "AsyncRT/ForwardDecls.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/Profiling/TimeProfiler.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>

namespace M::GML {
class Tensor;
}

namespace M {
class Model;
class ExecutorDriverOptions;

/// Profiler entry for main execute events.
using DriverProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kOther, 1), Trace::kOther>;

enum class InputGenerationKind {
  kRandom, // generate inputs with random elements, distribution depends on type
  kZeros,  // generate inputs filled with zeros
  kOnes,   // generate inputs filled with ones
  kNegOnes, // generate inputs filled with negative ones
};

StringLiteral stringifyEnum(InputGenerationKind kind);
std::optional<InputGenerationKind> symbolizeInputGenerationKind(StringRef s);

enum class TensorGeneratorFlag : int {
  kCycle = 1 << 0,    // loop over the file indefinitely
  kTruncate = 1 << 1, // discard a trailing partial tensor
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

/// Run `model` using the command line options flags configured by the user.
LogicalResult runModel(const Model *model, AsyncRT::CPUDevice &cpuDevice,
                       const ExecutorDriverOptions &options);

} // namespace M

#endif // GENERICML_EXECUTIONDRIVER_EXECUTIONDRIVER_H
