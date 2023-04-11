//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_DRIVERCLOPTIONS_H
#define GENERICML_DRIVER_DRIVERCLOPTIONS_H

#include "GenericML/Driver/DriverInputFormat.h"
#include "GenericML/Driver/ExecutionDriver.h"
#include "Kernels/Misc.h"
#include "LLCL/Runtime/RuntimeCLOptions.h"
#include "Support/CommonCLOptions.h"
#include "Support/ML/TensorShape.h"
#include "llvm/Support/NativeFormatting.h"
#include "llvm/Support/Process.h"

#include <filesystem>

namespace llvm {
class MemoryBuffer;
}

namespace M {

class TensorShapeParser : public llvm::cl::parser<TensorShape> {
public:
  using llvm::cl::parser<TensorShape>::parser;

  bool parse(llvm::cl::Option &O, StringRef ArgName, StringRef ArgValue,
             TensorShape &Val);
};

struct InputDataSpec {
  size_t tensorIdx;
  std::variant<InputGenerationKind, std::filesystem::path> inputGen;
  TensorGeneratorFlag genFlags;
};

class InputDataParser : public llvm::cl::parser<InputDataSpec> {
public:
  using llvm::cl::parser<InputDataSpec>::parser;

  bool parse(llvm::cl::Option &O, StringRef ArgName, StringRef ArgValue,
             InputDataSpec &Val) {
    if (auto errOr = parseImpl(O, ArgName, ArgValue, Val)) {
      llvm::errs() << errOr.getError() << "\n";
      return true;
    }
    return false;
  }

private:
  // Use this for cleaner error handling and then pass result to parse()
  ErrorOrSuccess parseImpl(llvm::cl::Option &O, StringRef ArgName,
                           StringRef ArgValue, InputDataSpec &Val);
};

/// Contains the core CLI options for *-executor, so that other executables
/// can use these without duplicating them.
class ExecutorDriverCLOptions : public LLCL::RuntimeCLOptions,
                                public CommonCLOptions {
public:
  using CommonCLOptions::CommonCLOptions;

  /// This specifies the number of times we should execute the model in a warmup
  /// phase. Warmup is only honored if the "num-runs" option is greater than 1
  /// (i.e. we are in a benchmarking phase).
  cl::opt<unsigned> numWarmup{
      "num-warmup",
      cl::desc("Execute inference model N times in warmup mode (default 0). "
               "This option is only used if \"num-runs\" > 1."),
      cl::init(0)};

  /// This specifies the number of times we should execute the model.
  cl::opt<unsigned> numRuns{
      "num-runs", cl::desc("Execute inference model N times (default 1)."),
      cl::init(1)};

  /// Benchmarks the model using the mlperf infrastructure.
  cl::opt<bool> benchmarkMode{
      "benchmark",
      cl::desc("If set, benchmarks the model using the mlperf infrastructure."),
      cl::init(false)};

  /// This specifies the name of the benchmark. Typically the benchmark name is
  /// the same as the model name.
  cl::opt<std::string> benchmarkName{"benchmark-name",
                                     cl::desc("The name of the benchmark."),
                                     cl::init("benchmark")};

  /// This specifies the benchmarking workload scenario during benchmarking.
  /// The valid options are "online" and "batched"
  cl::opt<BenchmarkScenario> benchmarkScenario{
      "benchmark-scenario", cl::desc("The benchmarking scenario to use."),
      cl::values(
          clEnumValN(
              BenchmarkScenario::kOnline, "online",
              "Online / streamed benchmarking execution mode. Corresponds "
              "to mlperf's 'single stream' model. Default."),
          clEnumValN(
              BenchmarkScenario::kBatched, "batched",
              "Batched benchmarking execution mode. Corresponds to mlperf's "
              "'multi stream' model).")),
      cl::init(BenchmarkScenario::kOnline)};

  /// This specifies the output directory which will contain the benchmarking
  /// logs. The default is to query the MODULAR_DERIVED_PATH env variable, if
  /// it is set then that's the default output directory. Otherwise, the current
  /// working directory is used.
  cl::opt<std::string> benchmarkOutputDir{
      "benchmark-output-dir",
      cl::desc("The output directory to place the benchmarking logs."),
      cl::init(
          llvm::sys::Process::GetEnv("MODULAR_DERIVED_PATH").value_or("."))};

  /// This specifies whether to prefix the benchmarking logs or not. If the
  /// option is not set, then the default prefix
  /// (modular_<benchmark_name>_benchmark_log_) is used
  cl::opt<bool> benchmarkNoLogPrefix{
      "no-benchmark-log-prefix",
      cl::desc(
          "Don't add prefix to mlperf results. If this option is not "
          "used the prefix will be modular_<benchmark_name>_benchmark_log_"),
      cl::init(false)};

  cl::opt<uint64_t> benchmarkMinDurationMs{
      "benchmark-min-duration-ms",
      cl::desc(
          "Minimum duration to run the benchmarks, in milliseconds. "
          "Corresponds to mlperf's min_duration_ms configuration parameter."),
      cl::init(10'000)};

  cl::opt<uint64_t> benchmarkMinQueryCount{
      "benchmark-min-query-count",
      cl::desc(
          "Minimum number of queries to run in benchmark mode. Corresponds to "
          "mlperf's min_query_count configuration parameter."),
      cl::init(100)};

  /// This tells the execution driver how to initialize model inputs.
  cl::opt<InputGenerationKind> inputGenKind{
      "model-inputs", cl::desc("Specify input tensor generation:"),
      cl::values(
          clEnumValN(
              InputGenerationKind::kRandom,
              stringifyEnum(InputGenerationKind::kRandom),
              "Input tensors are random (distribution is type dependent)"),
          clEnumValN(InputGenerationKind::kZeros,
                     stringifyEnum(InputGenerationKind::kZeros),
                     "All input tensors are zeros"),
          clEnumValN(InputGenerationKind::kNegOnes,
                     stringifyEnum(InputGenerationKind::kNegOnes),
                     "All input tensors are negatives ones"),
          clEnumValN(InputGenerationKind::kOnes,
                     stringifyEnum(InputGenerationKind::kOnes),
                     "All input tensors are ones")),
      cl::init(InputGenerationKind::kZeros)};

  cl::list<TensorShape, bool, TensorShapeParser> inputShapes{
      "input-shapes",
      cl::desc("Specify input shapes. Required when the model has "
               "dynamic input shapes. If specified and the model has static "
               "input shapes, it must match the model's input shapes. An "
               "empty value means scalar.")};

  cl::list<InputDataSpec, bool, InputDataParser> inputData{
      "input-data",
      cl::desc(
          "Specify custom data to use per input. It is specified as "
          "--input-data=tensor_idx:input_source:input_source_attr_list where:\n"
          "- tensor_idx: int representing the input tensor index\n"
          "- input_source: one of [random|zeros|negones|ones] or the "
          "name of a file containing raw binary tensor data stored in "
          "row-major little-endian order\n"
          "- input_source_attr_list: optional (default empty) comma separated "
          "list of [Truncate|Cycle] which specify relaxations "
          "on use of data in the input file. If empty, the file size must be "
          "an exact multiple of the input tensor size and every tensor in the "
          "input file may only be used once. If Cycle is specified, there is "
          "no limit on the number of times a tensor in the input file can be "
          "used. If Truncate is specified, the file size need not be an exact "
          "multiple of the tensor size (the remainder will be ignored).")};

  cl::opt<size_t> minimumCpuAlignment{
      "minimum-cpu-alignment",
      cl::desc(
          "The minimum byte alignment for all CPU-allocated buffers. Default "
          "is 64 bytes for compatability with the stock TensorFlow runtime."),
      cl::init(64)};

  /// This tells the execution driver how to print output tensors;
  cl::opt<Kernels::ResultOutputStyle> resultOutputStyle{
      "result-output-style", cl::desc("Specify how to display outputs"),
      cl::values(
          clEnumValN(Kernels::ResultOutputStyle::kFull, "full",
                     "Display all the elements of result"),
          clEnumValN(
              Kernels::ResultOutputStyle::kCompact, "compact",
              "Display a few elements at the start and end of each dimension"),
          clEnumValN(Kernels::ResultOutputStyle::kNone, "none",
                     "Do not display any result details")),
      cl::init(Kernels::ResultOutputStyle::kCompact)};

  /// If set, save the execution time of every non-warmup run to given file.
  cl::opt<std::string> saveTimings{
      "save-timings",
      cl::desc("Optional filename to capture the execution latency (in "
               "nanonseconds) per run. Use '-' to output to stdout. Does "
               "not include warmup runs. Ignored in --benchmark mode."),
      cl::init("")};
};

/// This command enum specifies what action we're supposed to take, which is
/// directly correlated with a level of abstraction.  Later stages do more
/// processing depending on the format of the input provided to the tool.
enum class DriverCommand {
  inspect,       // Print information about the input without processing it.
  printMetadata, // Print model information after conversion to MEF in YAML
                 // format. This command is used by other tools to introspect
                 // the model metadata.
  printConstantMetadata, // Print information on all the constants contained in
                         // YAML format.
  compile,               // Compile the input into a MEF file for execution.
  execute,               // Execute the input file.
  getVersion,    // Print the version string from the framework and exit.
  frameworkHelp, // Print the framework's help string.
};

/// This class provides the basic driver command options. It essentially exposes
/// the `DriverCommand` enum with some additional options.
class DriverCommandOptions {
public:
  cl::opt<DriverCommand> command{
      cl::desc("Command to execute"),
      cl::values(
          clEnumValN(DriverCommand::inspect, "inspect",
                     "Print information about the input without processing it"),
          clEnumValN(
              DriverCommand::printMetadata, "print-metadata",
              "Print model metadata in YAML format for other tools to use it"),
          clEnumValN(DriverCommand::printConstantMetadata,
                     "print-constant-metadata",
                     "Print constant metadata in YAML format for other tools "
                     "to use it"),
          clEnumValN(DriverCommand::compile, "compile",
                     "Compile the input into a MEF file for execution"),
          clEnumValN(DriverCommand::execute, "execute",
                     "Execute the input file"),
          clEnumValN(DriverCommand::getVersion, "get-version",
                     "Get the version of the loaded framework."),
          clEnumValN(DriverCommand::frameworkHelp, "framework-help",
                     "Get the framework help string.")),
      llvm::cl::Required};

  /// Return true if the selected output format is binary and shouldn't be
  /// dumped to the console.
  bool commandHasBinaryOutput() const {
    return command == DriverCommand::compile;
  }

  /// Return true if the command passed in requires an input file.
  bool commandTakesInput() const {
    return command != DriverCommand::getVersion &&
           command != DriverCommand::frameworkHelp;
  }
};

/// This class provides essentially all the command-line options available to
/// the driver. It amalgamates the commands from DriverCommandOptions and
/// ExecutorDriverCLOptions, and provides input format and output file
/// specification. It also provides utility methods for opening an
/// output/intermediate file with a given filename.
class DriverCLOptions : public DriverInputFormatOptions,
                        public ExecutorDriverCLOptions {
public:
  using ExecutorDriverCLOptions::ExecutorDriverCLOptions;

  //===--------------------------------------------------------------------===//
  // MEF Emission Configuration
  //===--------------------------------------------------------------------===//

  cl::opt<bool> enableMEFMetadata{
      "enable-mef-metadata",
      cl::desc("Enable metadata sections for better MEF2MLIR"),
      cl::init(false)};
};

/// This class provides additional command line options for the
/// tfl-graph-verifier, namely the tolerances (rtol and atol) which are used to
/// determine whether there is an accuracy discrepancy between the TFLP and
/// FlatBuffer outputs.
class VerifyGraphDriverCLOptions : public ExecutorDriverCLOptions {
public:
  using ExecutorDriverCLOptions::ExecutorDriverCLOptions;

  cl::opt<double> atol{
      "atol",
      cl::desc("Absolute tolerance for determining whether there's an accuracy "
               "discrepancy between TFLP and stock-tflite."),
      cl::init(1.0E-5)};

  cl::opt<double> rtol{
      "rtol",
      cl::desc("Relative tolerance for determining whether there's an "
               "accuracy discrepancy between TFLP and stock-tflite."),
      cl::init(1.0E-8)};
};

} // namespace M

#endif // GENERICML_DRIVER_DRIVERCLOPTIONS_H
