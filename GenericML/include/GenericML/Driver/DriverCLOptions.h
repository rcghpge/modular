//===- GenericML/Driver/DriverCLOptions.h ---------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_DRIVERCLOPTIONS_H
#define GENERICML_DRIVER_DRIVERCLOPTIONS_H

#include "GenericML/Driver/DriverInputFormat.h"
#include "GenericML/Driver/ExecutionDriver.h"
#include "LLCL/Runtime/RuntimeCLOptions.h"
#include "Support/CommonCLOptions.h"
#include "llvm/Support/NativeFormatting.h"

namespace llvm {
class MemoryBuffer;
}

namespace M {

/// Contains the core CLI options for *-executor, so that other executables
/// can use these without duplicating them.
class ExecutorDriverCLOptions : public LLCL::RuntimeCLOptions,
                                public CommonCLOptions {
public:
  ExecutorDriverCLOptions(StringRef programName)
      : CommonCLOptions(programName) {}

  /// This specifies the number of times we should execute the model in a warmup
  /// phase. Warmup is only honored if the "num-runs" option is greater than 1
  /// (i.e. we are in a benchmarking phase).
  cl::opt<unsigned> numWarmup{
      "num-warmup",
      cl::desc("Execute inference model N times in warmup mode (default: 1). "
               "This option is only used if \"num-runs\" > 1."),
      cl::init(1)};

  /// This specifies the number of times we should execute the model.  If run
  /// more than one time, we enter "benchmarking" mode.
  cl::opt<unsigned> numRuns{
      "num-runs", cl::desc("Execute inference model N times (default: 1)."),
      cl::init(1)};

  /// This specifies the number of times we should execute the model.  If run
  /// more than one time, we enter "benchmarking" mode.
  cl::opt<bool> benchmarkMode{
      "benchmark", cl::desc("Measure each invocation of the model inference."),
      cl::init(false)};

  /// This specifies the name of the benchmark. Typically the benchmark name is
  /// the same as the model name.
  cl::opt<std::string> benchmarkName{"benchmark-name",
                                     cl::desc("The name of the benchmark."),
                                     cl::init("benchmark")};

  /// This specifies the benchmarking workload scenario during benchmarking.
  /// The valid options are "online" and "batched"
  cl::opt<BenchmarkScenario> benchmarkScenario{
      "benchmark-scenario", cl::desc("The benchmarking scenario used."),
      cl::values(clEnumValN(BenchmarkScenario::kOnline, "online",
                            "Online / streamed input generation"),
                 clEnumValN(BenchmarkScenario::kBatched, "batched",
                            "Batched generation")),
      cl::init(BenchmarkScenario::kOnline)};

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

  /// This tells the execution driver how to print output tensors;
  cl::opt<Kernel::ResultOutputStyle> resultOutputStyle{
      "result-output-style", cl::desc("Specify how to display outputs"),
      cl::values(
          clEnumValN(Kernel::ResultOutputStyle::kFull, "full",
                     "Display all the elements of result"),
          clEnumValN(
              Kernel::ResultOutputStyle::kCompact, "compact",
              "Display a few elements at the start and end of each dimension")),
      cl::init(Kernel::ResultOutputStyle::kCompact)};
};

/// This command enum specifies what action we're supposed to take, which is
/// directly correlated with a level of abstraction.  Later stages do more
/// processing depending on the format of the input provided to the tool.
enum class DriverCommand {
  inspect,       // Print information about the input without processing it.
  printMetadata, // Print model information after conversion to BEF in YAML
                 // format. This command is used by other tools to introspect
                 // the model metadata.
  printConstantMetadata, // Print information on all the constants contained in
                         // YAML format.
  compile,               // Compile the input into a BEF file for execution.
  execute,               // Execute the input file.
  getVersion, // Print the version string from the framework and exit.
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
                     "Compile the input into a BEF file for execution"),
          clEnumValN(DriverCommand::execute, "execute",
                     "Execute the input file"),
          clEnumValN(DriverCommand::getVersion, "get-version",
                     "Get the version of the loaded framework.")),
      llvm::cl::Required};

  cl::opt<bool> saveTemps{
      "save-temps",
      cl::desc(
          "Store the usual 'temporary' intermediate files permanently in "
          "the directory specified by -temps-dir (defaults to the output "
          "directory); name them as auxiliary output files. This flag only "
          "changes the behavior of the `compile` and `execute` commands."),
      llvm::cl::Optional};

  cl::opt<std::string> tempsDir{
      "temps-dir", cl::init(""),
      cl::desc(
          "The directory in which to store 'temporary' intermediate files. No "
          "files will be saved here unless `-save-temps` is also specified."),
      llvm::cl::Optional};

  /// Return true if the selected output format is binary and shouldn't be
  /// dumped to the console.
  bool commandHasBinaryOutput() const {
    return command == DriverCommand::compile;
  }
};

/// This class provides essentially all the command-line options available to
/// the driver. It amalgamates the commands from DriverCommandOptions and
/// ExecutorDriverCLOptions, and provides input format and output file
/// specification. It also provides utility methods for opening an
/// output/intermediate file with a given filename.
class DriverCLOptions : public DriverCommandOptions,
                        public ExecutorDriverCLOptions {
public:
  DriverCLOptions(StringRef programName)
      : ExecutorDriverCLOptions(programName) {}
  /// Allow the user to override the input file format.  The default is to
  /// autosense what the input is.
  cl::opt<FileFormat> inputFormat{
      "input-format", cl::desc("input file format override"),
      cl::values(clEnumValN(FileFormat::kTFLFlatBuffer, "tflite-flat-buffer",
                            "Input is a TFLite flatbuffer file"),
                 clEnumValN(FileFormat::kBEFFile, "bef",
                            "Input is an executable BEF file")),
      cl::init(FileFormat::kUnknown)};

  /// Return the input format the user specified, allowing them to specify
  /// something manually on the command line, but otherwise autosensing an
  /// answer.
  FileFormat getInputFormat(StringRef fileContents) const;

  //===--------------------------------------------------------------------===//
  // Force loading a specific dylib
  //===--------------------------------------------------------------------===//

  cl::opt<std::string> loadLib{
      "load", cl::desc("Load the specified framework directly."), cl::init("")};

  //===--------------------------------------------------------------------===//
  // Emission Options
  //===--------------------------------------------------------------------===//

  cl::opt<std::string> outputFilename{"o", cl::desc("Output filename"),
                                      cl::value_desc("filename"),
                                      cl::init("-")};

  /// Determine an output file name and open it.
  std::unique_ptr<llvm::ToolOutputFile> getOutputFile() const;

  /// Determine an intermediate file with extension `ext` and open it.
  std::unique_ptr<llvm::ToolOutputFile>
  getIntermediateFile(const std::string &ext) const;

  //===--------------------------------------------------------------------===//
  // BEF Emission Configuration
  //===--------------------------------------------------------------------===//

  cl::opt<bool> enableBEFMetadata{
      "enable-bef-metadata",
      cl::desc("Enable metadata sections for better BEF2MLIR"),
      cl::init(false)};
};

} // namespace M

#endif // GENERICML_DRIVER_DRIVERCLOPTIONS_H
