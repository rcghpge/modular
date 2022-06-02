//===- GenericML/ExecutionDriverOptions.h ---------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_EXECUTIONDRIVEROPTIONS_H
#define GENERICML_EXECUTIONDRIVEROPTIONS_H

#include "GenericML/ExecutionDriver.h"
#include "GraphRT/Tools/CLOptions.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

namespace M {

/// Contains the core CLI options for *-executor, so that other executables
/// can use these without duplicating them.
struct ExecutorDriverCLOptions : public BEFCLOptionsCore, public CLOptionsCore {
  /// This specifies the number of times we should execute the model.  If run
  /// more than one time, we enter "benchmarking" mode.
  cl::opt<unsigned> numRuns{
      "num-runs", cl::desc("Execute inference model N times (default: 1)."),
      cl::init(1)};

  /// This specifies the number of times we should execute the model.  If run
  /// more than one time, we enter "benchmarking" mode.
  cl::opt<bool> benchmarkMode{
      "benchmark", cl::desc("Measure each invocation of the model inference"),
      cl::init(false)};

  /// This tells the execution driver how to initialize model inputs.
  cl::opt<InputGenerationKind> inputGenKind{
      "model-inputs", cl::desc("Specify input tensor generation:"),
      cl::values(
          clEnumValN(
              InputGenerationKind::kRandom, "random",
              "Input tensors are random (distribution is type dependent)"),
          clEnumValN(InputGenerationKind::kZeros, "zeros",
                     "All input tensors are zeros"),
          clEnumValN(InputGenerationKind::kNegOnes, "negones",
                     "All input tensors are negatives ones"),
          clEnumValN(InputGenerationKind::kOnes, "ones",
                     "All input tensors are ones")),
      cl::init(InputGenerationKind::kZeros)};
};

} // namespace M

#endif // GENERICML_EXECUTIONDRIVEROPTIONS_H
