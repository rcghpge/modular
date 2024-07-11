//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_DRIVERCLOPTIONS_H
#define GENERICML_DRIVER_DRIVERCLOPTIONS_H

#include "ArraySupport/Tensor.h"
#include "AsyncRT/Runtime/RuntimeCLOptions.h"
#include "GenericML/Driver/DriverInputFormat.h"
#include "GenericML/Driver/ExecutionDriver.h"
#include "Support/CommonCLOptions.h"
#include "Support/ML/TensorSpec.h"
#include "llvm/Support/NativeFormatting.h"
#include "llvm/Support/Process.h"

#include <filesystem>
#include <variant>

namespace llvm {
class MemoryBuffer;
}

namespace M {

class TensorSpecParser : public llvm::cl::parser<TensorSpec> {
public:
  using llvm::cl::parser<TensorSpec>::parser;

  bool parse(llvm::cl::Option &options, StringRef argName, StringRef argValue,
             TensorSpec &result);
};
class TensorShapeParser : public llvm::cl::parser<TensorShape> {
public:
  using llvm::cl::parser<TensorShape>::parser;

  bool parse(llvm::cl::Option &options, StringRef argName, StringRef argValue,
             TensorShape &result);
};

struct InputDataSpec {
  size_t tensorIdx;
  std::variant<InputGenerationKind, std::filesystem::path> inputGen;
  TensorGeneratorFlag genFlags;
};

class InputDataParser : public llvm::cl::parser<InputDataSpec> {
public:
  using llvm::cl::parser<InputDataSpec>::parser;

  bool parse(llvm::cl::Option &options, StringRef argName, StringRef argValue,
             InputDataSpec &result) {
    if (auto errOr = parseImpl(options, argName, argValue, result)) {
      llvm::errs() << errOr.getError() << "\n";
      return true;
    }
    return false;
  }

private:
  // Use this for cleaner error handling and then pass result to parse()
  ErrorOrSuccess parseImpl(llvm::cl::Option &options, StringRef argName,
                           StringRef argValue, InputDataSpec &result);
};

/// Contains the core CLI options for *-executor, so that other executables
/// can use these without duplicating them.
class ExecutorDriverOptions : public LLCL::RuntimeOptions,
                              public CommonOptions {

public:
  //   using CommonCLOptions::CommonCLOptions;

  /// This specifies the number of times we should execute the model in a warmup
  /// phase. Warmup is only honored if the "num-runs" option is greater than 1
  /// (i.e. we are in a benchmarking phase).
  unsigned numWarmup{0};

  /// This specifies the number of times we should execute the model.
  unsigned numRuns{1};

  /// Benchmarks the model using the mlperf infrastructure.
  bool benchmarkMode{false};

  /// This specifies the name of the benchmark. Typically the benchmark name is
  /// the same as the model name.
  std::string benchmarkName{"benchmark"};

  /// This specifies the benchmarking workload scenario during benchmarking.
  /// The valid options are "online" and "batched"
  BenchmarkScenario benchmarkScenario{BenchmarkScenario::kOnline};

  /// This specifies the output directory which will contain the benchmarking
  /// logs. The default is to query the MODULAR_DERIVED_PATH env variable, if
  /// it is set then that's the default output directory. Otherwise, the current
  /// working directory is used.
  std::string benchmarkOutputDir{
      llvm::sys::Process::GetEnv("MODULAR_DERIVED_PATH").value_or(".")};

  /// This specifies whether to prefix the benchmarking logs or not. If the
  /// option is not set, then the default prefix
  /// (modular_<benchmark_name>_benchmark_log_) is used
  bool benchmarkNoLogPrefix{false};

  uint64_t benchmarkMinDurationMs{10'000};

  uint64_t benchmarkMinQueryCount{100};

  /// This tells the execution driver how to initialize model inputs.
  InputGenerationKind inputGenKind{InputGenerationKind::kZeros};

  SmallVector<TensorSpec> inputSpecs{};

  SmallVector<TensorShape> inputShapes{};

  SmallVector<InputDataSpec> inputData{};

  size_t minimumCpuAlignment{64};

  /// This tells the execution driver how to print output tensors;
  ResultOutputStyle resultOutputStyle{ResultOutputStyle::kCompact};

  unsigned printPrecision{6};

  std::string binaryOutputDir{""};

  /// If set, save the execution time of every non-warmup run to given file.
  std::string saveTimings{""};
};

/// Contains the core CLI options for *-executor, so that other executables
/// can use these without duplicating them.
class ExecutorDriverCLOptions : public LLCL::RuntimeCLOptions,
                                public CommonCLOptions {
public:
  ExecutorDriverOptions &options;
  using CommonCLOptions::CommonCLOptions;
  ExecutorDriverCLOptions(int argc, char **argv, ExecutorDriverOptions &opts,
                          bool skipInitLLVM = false)
      : RuntimeCLOptions(opts), CommonCLOptions(argc, argv, opts, skipInitLLVM),
        options(opts) {}

private:
  llvm::cl::OptionCategory ExecutorDriverOptionsCategory{
      "Executor driver command line options"};
  /// This specifies the number of times we should execute the model in a warmup
  /// phase. Warmup is only honored if the "num-runs" option is greater than 1
  /// (i.e. we are in a benchmarking phase).
  M::cl::MOpt<unsigned, true> numWarmup{
      "num-warmup",
      cl::desc("Execute inference model N times in warmup mode (default 0). "
               "This option is only used if \"num-runs\" > 1."),
      llvm::cl::location(options.numWarmup),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  /// This specifies the number of times we should execute the model.
  M::cl::MOpt<unsigned, true> numRuns{
      "num-runs", cl::desc("Execute inference model N times (default 1)."),
      llvm::cl::location(options.numRuns),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  /// Benchmarks the model using the mlperf infrastructure.
  M::cl::MOpt<bool, true> benchmarkMode{
      "benchmark",
      cl::desc("If set, benchmarks the model using the mlperf infrastructure."),
      llvm::cl::location(options.benchmarkMode),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  /// This specifies the name of the benchmark. Typically the benchmark name is
  /// the same as the model name.
  M::cl::MOpt<std::string, true> benchmarkName{
      "benchmark-name", cl::desc("The name of the benchmark."),
      llvm::cl::location(options.benchmarkName),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  /// This specifies the benchmarking workload scenario during benchmarking.
  /// The valid options are "online" and "batched"
  M::cl::MOpt<BenchmarkScenario, true> benchmarkScenario{
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
      llvm::cl::location(options.benchmarkScenario),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  /// This specifies the output directory which will contain the benchmarking
  /// logs. The default is to query the MODULAR_DERIVED_PATH env variable, if
  /// it is set then that's the default output directory. Otherwise, the current
  /// working directory is used.
  M::cl::MOpt<std::string, true> benchmarkOutputDir{
      "benchmark-output-dir",
      cl::desc("The output directory to place the benchmarking logs."),
      llvm::cl::location(options.benchmarkOutputDir),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  /// This specifies whether to prefix the benchmarking logs or not. If the
  /// option is not set, then the default prefix
  /// (modular_<benchmark_name>_benchmark_log_) is used
  M::cl::MOpt<bool, true> benchmarkNoLogPrefix{
      "no-benchmark-log-prefix",
      cl::desc(
          "Don't add prefix to mlperf results. If this option is not "
          "used the prefix will be modular_<benchmark_name>_benchmark_log_"),
      llvm::cl::location(options.benchmarkNoLogPrefix),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  M::cl::MOpt<uint64_t, true> benchmarkMinDurationMs{
      "benchmark-min-duration-ms",
      cl::desc(
          "Minimum duration to run the benchmarks, in milliseconds. "
          "Corresponds to mlperf's min_duration_ms configuration parameter."),
      llvm::cl::location(options.benchmarkMinDurationMs),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  M::cl::MOpt<uint64_t, true> benchmarkMinQueryCount{
      "benchmark-min-query-count",
      cl::desc(
          "Minimum number of queries to run in benchmark mode. Corresponds to "
          "mlperf's min_query_count configuration parameter."),
      llvm::cl::location(options.benchmarkMinQueryCount),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  /// This tells the execution driver how to initialize model inputs.
  M::cl::MOpt<InputGenerationKind, true> inputGenKind{
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
      llvm::cl::location(options.inputGenKind),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  M::cl::MListOpt<TensorSpec, SmallVector<TensorSpec>, TensorSpecParser>
      inputSpecs{
          "input-specs",
          cl::desc("Specify input specifications. Required when the model has "
                   "no or partially fixed shapes or data types. Input shapes "
                   "('--input-shapes') can be provided additionally, but must "
                   "match the"
                   "model's input specs."),
          llvm::cl::location(options.inputSpecs),
          llvm::cl::cat(ExecutorDriverOptionsCategory)};

  M::cl::MListOpt<TensorShape, SmallVector<TensorShape>, TensorShapeParser>
      inputShapes{
          "input-shapes",
          cl::desc(
              "Specify input shapes. Required when the model has "
              "dynamic input shapes. If specified and the model has static "
              "input shapes, it must match the model's input shapes. An "
              "empty value means scalar."),
          llvm::cl::location(options.inputShapes),
          llvm::cl::cat(ExecutorDriverOptionsCategory)};

  M::cl::MListOpt<InputDataSpec, SmallVector<InputDataSpec>, InputDataParser>
      inputData{
          "input-data",
          cl::desc(
              "Specify custom data to use per input. It is specified as "
              "--input-data=tensor_idx::input_source::input_source_attr_list "
              "where:\n"
              "- tensor_idx: int representing the input tensor index\n"
              "- input_source: one of [random|zeros|negones|ones] or "
              "the name of a file containing raw binary tensor data stored in "
              "row-major little-endian order\n"
              "- input_source_attr_list: "
              "optional (default empty) comma separated list of "
              "[Truncate|Cycle] which specify relaxations on use of data in "
              "the input file. If empty, the file size must be an exact "
              "multiple of the input tensor size and every tensor in the input "
              "file may only be used once. If Cycle is specified, there is no "
              "limit on the number of times a tensor in the input file can be "
              "used. If Truncate is specified, the file size need not be an "
              "exact multiple of the tensor size (the remainder will be "
              "ignored)."),
          llvm::cl::location(options.inputData),
          llvm::cl::cat(ExecutorDriverOptionsCategory)};

  M::cl::MOpt<size_t, true> minimumCpuAlignment{
      "minimum-cpu-alignment",
      cl::desc(
          "The minimum byte alignment for all CPU-allocated buffers. Default "
          "is 64 bytes for compatability with the stock TensorFlow runtime."),
      llvm::cl::location(options.minimumCpuAlignment),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  /// This tells the execution driver how to print output tensors;
  M::cl::MOpt<ResultOutputStyle, true> resultOutputStyle{
      "result-output-style", cl::desc("Specify how to display outputs"),
      cl::values(
          clEnumValN(ResultOutputStyle::kFull, "full",
                     "Display all the elements of result"),
          clEnumValN(
              ResultOutputStyle::kCompact, "compact",
              "Display a few elements at the start and end of each dimension"),
          clEnumValN(
              ResultOutputStyle::kBinary, "binary",
              "Save outputs to binary files. This creates a directory "
              "with one binary file for each output tensor and a YAML file "
              "containing the name and shape of each output tensor. The "
              "output directory containing these files must be "
              "specified with --binary-output-dir=<output_dir>."),
          clEnumValN(ResultOutputStyle::kNone, "none",
                     "Do not display any result details")),
      llvm::cl::location(options.resultOutputStyle),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  M::cl::MOpt<unsigned, true> printPrecision{
      "print-precision",
      cl::desc("Specify precision of floating-point outputs (full-mode only)"),
      llvm::cl::location(options.printPrecision),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  M::cl::MOpt<std::string, true> binaryOutputDir{
      "binary-output-dir",
      cl::desc(
          "If --result-output-style=binary, specify the output directory."),
      llvm::cl::location(options.binaryOutputDir),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  /// If set, save the execution time of every non-warmup run to given file.
  M::cl::MOpt<std::string, true> saveTimings{
      "save-timings",
      cl::desc("Optional filename to capture the execution latency (in "
               "nanonseconds) per run. Use '-' to output to stdout. Does "
               "not include warmup runs. Ignored in --benchmark mode."),
      llvm::cl::location(options.saveTimings),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};
};

/// This command enum specifies what action we're supposed to take, which is
/// directly correlated with a level of abstraction.  Later stages do more
/// processing depending on the format of the input provided to the tool.
enum class DriverCommand {
  inspect,       // Print information about the input without processing it.
  printMetadata, // Print model information after conversion to MEF in YAML
                 // format. This command is used by other tools to introspect
                 // the model metadata.
  printConstantMetadata, // Print information on all the constants contained
                         // in YAML format.
  compile,               // Compile the input into a MEF file for execution.
  execute,               // Execute the input file.
  getVersion,    // Print the version string from the framework and exit.
  frameworkHelp, // Print the framework's help string.
};

/// This class provides the basic driver command options. It essentially
/// exposes the `DriverCommand` enum with some additional options.

class DriverCommandOptions {

public:
  DriverCommand command{};

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
class DriverCommandCLOptions {
private:
  llvm::cl::OptionCategory DriverCommandOptionsCategory{
      "Common driver commands"};

public:
  DriverCommandOptions &options;
  DriverCommandCLOptions(DriverCommandOptions &opts) : options(opts) {}
  M::cl::MOpt<DriverCommand, true> command{
      cl::desc("Command to execute"),
      cl::values(
          clEnumValN(DriverCommand::inspect, "inspect",
                     "Print information about the input without processing it"),
          clEnumValN(DriverCommand::compile, "compile",
                     "Compile the input into a MEF file for execution"),
          clEnumValN(DriverCommand::execute, "execute",
                     "Execute the input file"),
          clEnumValN(DriverCommand::getVersion, "get-version",
                     "Get the version of the loaded framework."),
          clEnumValN(DriverCommand::frameworkHelp, "framework-help",
                     "Get the framework help string.")),
      llvm::cl::Required, llvm::cl::cat(DriverCommandOptionsCategory),
      llvm::cl::location(options.command)};
};

/// This class provides essentially all the command-line options available to
/// the driver. It amalgamates the commands from DriverCommandOptions and
/// ExecutorDriverCLOptions, and provides input format and output file
/// specification. It also provides utility methods for opening an
/// output/intermediate file with a given filename.
class DriverOptions : public DriverInputFormatOptions,
                      public ExecutorDriverOptions {
public:
  //===--------------------------------------------------------------------===//
  // MEF Emission Configuration
  //===--------------------------------------------------------------------===//

  bool enableMEFMetadata{false};
};
class DriverCLOptions : public DriverInputFormatCLOptions,
                        public ExecutorDriverCLOptions {
public:
  using ExecutorDriverCLOptions::ExecutorDriverCLOptions;
  DriverOptions &options;
  DriverCLOptions(int argc, char **argv, DriverOptions &opts,
                  bool skipInitLLVM = false)
      : DriverInputFormatCLOptions(opts),
        ExecutorDriverCLOptions(argc, argv, opts, skipInitLLVM), options(opts) {
  }
  //===--------------------------------------------------------------------===//
  // MEF Emission Configuration
  //===--------------------------------------------------------------------===//

  M::cl::MOpt<bool, true> enableMEFMetadata{
      "enable-mef-metadata",
      cl::desc("Enable metadata sections for better MEF2MLIR"),
      llvm::cl::location(options.enableMEFMetadata)};
};

/// This class provides additional command line options for the
/// tfl-graph-verifier, namely the tolerances (rtol and atol) which are used
/// to determine whether there is an accuracy discrepancy between the TFLP and
/// FlatBuffer outputs.

class VerifyGraphDriverOptions : public ExecutorDriverOptions {

public:
  double atol{1.0E-5};
  double rtol{1.0E-8};
};
class VerifyGraphDriverCLOptions : public ExecutorDriverCLOptions {

public:
  VerifyGraphDriverOptions &options;
  VerifyGraphDriverCLOptions(int argc, char **argv,
                             VerifyGraphDriverOptions &opts,
                             bool skipInitLLVM = false)
      : ExecutorDriverCLOptions(argc, argv, opts, skipInitLLVM), options(opts) {
  }

private:
  llvm::cl::OptionCategory VerifyGraphDriverCLOptionsCategory{
      "Graph verification options"};

  M::cl::MOpt<double, true> atol{
      "atol",
      cl::desc("Absolute tolerance for determining whether there's an accuracy "
               "discrepancy between TFLP and stock-tflite."),
      llvm::cl::location(options.atol),
      llvm::cl::cat(VerifyGraphDriverCLOptionsCategory)};

  M::cl::MOpt<double, true> rtol{
      "rtol",
      cl::desc("Relative tolerance for determining whether there's an "
               "accuracy discrepancy between TFLP and stock-tflite."),
      llvm::cl::location(options.rtol),
      llvm::cl::cat(VerifyGraphDriverCLOptionsCategory)};
};

} // namespace M

#endif // GENERICML_DRIVER_DRIVERCLOPTIONS_H
