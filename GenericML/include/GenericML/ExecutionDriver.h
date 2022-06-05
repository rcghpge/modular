//===- GenericML/ExecutionDriver.h ----------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides an interface to the standardized "frwk-execute" command
// line tools.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_EXECUTIONDRIVER_H
#define GENERICML_EXECUTIONDRIVER_H

#include "LLCL/ForwardDecls.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include <cstddef>

namespace M {
class Model;
class Tensor;

enum class InputGenerationKind {
  kRandom, // generate inputs with random elements, distribution depends on type
  kZeros,  // generate inputs filled with zeros
  kOnes,   // generate inputs filled with ones
  kNegOnes, // generate inputs filled with negative ones
};

enum class BenchmarkScenario { kOnline, kBatched };

/// Initialize raw pointers for each model input with the correct data fill.
/// This allocates the pointer, fills in the data, and returns ownership to the
/// caller.
ErrorOr<SmallVector<Tensor, 3>>
initializeInputMemory(InputGenerationKind inputGenKind, const Model *model,
                      Runtime &runtime);

/// Run `model` using `runtime` `numRuns` times.
void runModel(const Model *model, LLCL::Runtime &runtime,
              InputGenerationKind inputGenKind, size_t numRuns = 1);

LogicalResult benchmarkModel(const Model *model, Runtime &runtime,
                             StringRef benchmarkName,
                             InputGenerationKind inputGenKind,
                             BenchmarkScenario scenario, StringRef outputDir,
                             bool benchmarkNoLogPrefix);

} // namespace M

#endif // GENERICML_EXECUTIONDRIVER_H
