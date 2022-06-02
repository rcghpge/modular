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

  /// This specifies the name of the benchmark. Typically the benchmark name is
  /// the same as the model name.
  cl::opt<std::string> benchmarkName{"benchmark-name",
                                     cl::desc("The name of the benchmark."),
                                     cl::init("modular")};

  /// This specifies the benchmarking workload scenario during benchmarking.
  /// The valid options are "online" and "batched"
  cl::opt<std::string> benchmarkScenario{
      "benchmark-scenario", cl::desc("The benchmarking scenario used."),
      cl::init("online")};

  /// This specifies the output directory which will contain the benchmarking
  /// logs
  cl::opt<std::string> benchmarkOutputDir{
      "benchmark-output-dir",
      cl::desc("The output directory to place the benchmarking logs."),
      cl::init(".")};

  /// This specifies whether to prefix the benchmarking logs or not. If the
  /// option is not set, then the default prefix
  /// (modular_<benchmark_name>_benchmark_log_) is used
  cl::opt<bool> benchmarkNoLogPrefix{
      "no-benchmark-log-prefix",
      cl::desc(
          "Don't add prefix to mlperf results. If this option is not "
          "used the prefix will be modular_<benchmark_name>_benchmark_log_"),
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
