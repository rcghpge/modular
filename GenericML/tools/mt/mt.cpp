//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Config/Version.h"
#include "GenericML/Driver/CXXDriverAPI.h"
#include "GenericML/Driver/DriverAPI.h"
#include "GenericML/Driver/DriverInputFormat.h"
#include "Support/ML/SizeUtils.h"
#include "llvm/Support/ToolOutputFile.h"

#include <filesystem>

using namespace M;
using llvm::MemoryBuffer;

namespace {
//===--------------------------------------------------------------------===//
// Scenario structure and flag parsing
//===--------------------------------------------------------------------===//

/// This struct provides the backing data store behind a "scenario". It
/// describes a command-line flag that takes one of 2 forms.
///
/// --scenario="--num-threads=3 --unchain=true"
/// OR
/// --scenario="nofail:--num-threads=3 --unchain=true"
///
/// The first option indicates that `mt` should abort on failure of the
/// scenario, while the second indicates that `mt` should continue even if the
/// scenario fails.
struct Scenario {
  bool abortOnFailure = true;
  SmallVector<std::string> flags = {};
};

class ScenarioParser : public llvm::cl::parser<Scenario> {
public:
  using llvm::cl::parser<Scenario>::parser;

  bool parse(llvm::cl::Option &O, StringRef ArgName, StringRef ArgValue,
             Scenario &Val);
};

//===--------------------------------------------------------------------===//
// GenericDriverCLOptions
//===--------------------------------------------------------------------===//

class GenericDriverCLOptions : public DriverCommandOptions,
                               public DriverInputFormatOptions,
                               public CommonCLOptions {
public:
  using CommonCLOptions::CommonCLOptions;

  //===--------------------------------------------------------------------===//
  // Force loading a specific dylib
  //===--------------------------------------------------------------------===//

  cl::opt<std::string> loadLib{
      "load", cl::desc("Load the specified framework directly."), cl::init("")};

  //===--------------------------------------------------------------------===//
  // Specify a 'scenario' to run.
  //===--------------------------------------------------------------------===//

  cl::list<Scenario, bool, ScenarioParser> scenarios{
      "scenario",
      cl::desc("List of scenarios to run. Only has meaning with the -execute "
               "command. Each scenario's strings are passed directly to the "
               "loaded framework."),
  };

  //===--------------------------------------------------------------------===//
  // Sink for all unparsed options.
  //===--------------------------------------------------------------------===//

  cl::list<std::string> forceUnparsed{
      "X", cl::desc("Arguments to be passed to the framework.")};

  cl::list<std::string> unparsed{
      llvm::cl::Sink,
      cl::desc("Arguments to be passed directly to the framework."),
  };
};
} // namespace

//===--------------------------------------------------------------------===//
// ScenarioParser implementation
//===--------------------------------------------------------------------===//

bool ScenarioParser::parse(llvm::cl::Option &O, StringRef ArgName,
                           StringRef ArgValue, Scenario &Val) {
  if (ArgValue.consume_front("nofail:"))
    Val.abortOnFailure = false;

  // The flags are what remains.
  llvm::transform(llvm::split(ArgValue, ' '), std::back_inserter(Val.flags),
                  [](StringRef s) { return s.str(); });
  return false;
}

/// Get a string that describes the contents of the file. Given that this driver
/// cannot link the MEF libraries, we have to rely on the symbols provided by
/// the framework if the input is a MEF file.
static M::ErrorOr<std::string>
getFileContentIdentity(const llvm::MemoryBuffer &file, FileFormat format,
                       FrameworkContextHandle ctxHandle, CXXDriverAPI dylib) {
  switch (format) {
  default:
    return "input has unknown contents";
  case FileFormat::kTFLFlatBuffer:
    return "input is a binary TFLite flatbuffer file";
  case FileFormat::kTFLFlatBufferJSON:
    return "input is a TFLite flatbuffer file in JSON format";
  case FileFormat::kMTFLDialect:
    return "input is TFLite model in MLIR form";
  case FileFormat::kTFLPDialect:
    return "input is a compiled TFLite MEF model in MLIR form";
  case FileFormat::kTFDialect:
    return "input is a TensorFlow model in MLIR form";
  case FileFormat::kTFSavedModel:
    return "input is a TensorFlow model in SavedModel form";
  case FileFormat::kTFPDialect:
    return "input is a compiled TensorFlow MEF model in MLIR form";
  case FileFormat::kFauxDialect:
    return "input is Faux model in MLIR form";
  case FileFormat::kMONNXDialect:
    return "input is an ONNX model in MLIR form";
  case FileFormat::kONNXProtobuf:
    return "input is an ONNX model in binary protobuf form";
  case FileFormat::kONNXTextProtobuf:
    return "input is an ONNX model in text protobuf form";
  case FileFormat::kTorchDialect:
    return "input is a PyTorch model in Torch MLIR dialect form";
  case FileFormat::kPTPickle:
    return "input is a PyTorch model in PT pickled form";
  case FileFormat::kMEFFile: {
    UNWRAP_ERROR(fileHandle, dylib.openBinaryFile(ctxHandle, file));
    return dylib.sniffBinaryFileContents(ctxHandle, fileHandle);
  }
  }
}

int main(int argc, char **argv) {
  // Register all our command line options.  We expect that the subframeworks
  // will not call InitLLVM, and that is the responsibility of the driver.
  GenericDriverCLOptions options(argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv, "Modular Tool (mt)\n");

  struct InputInfo {
    // If we don't have a regular input file (and we're attempting to run
    // something that doesn't take any input), then we have to pass in an empty
    // memory buffer which will result in an unknown file format. We want to
    // preserve the stdin behavior for cases where the command does take input
    // because we might be piping something into the tool.
    std::unique_ptr<MemoryBuffer> file =
        MemoryBuffer::getMemBuffer("", "empty");

    FileFormat format = FileFormat::kUnknown;
    bool isInputDirectory = false;
    std::filesystem::path inputPath;
  };

  std::vector<InputInfo> inputsInfo{};

  // Handle stdin specially by default-constructing an `InputInfo` so we can
  // still process the commands later on.
  auto fromStdin = options.inputFilename == "-";
  if (fromStdin) {
    inputsInfo.emplace_back();
  }

  if (options.commandTakesInput() || !fromStdin) {
    // Input filenames are comma-separated (e.g. file1,file2).  In the future,
    // we may change the CL option to be a list<string> instead.
    auto inputAdaptorRange = llvm::split(options.inputFilename, ",");
    if (std::distance(inputAdaptorRange.begin(), inputAdaptorRange.end()) > 1 &&
        options.command != DriverCommand::execute) {
      return options.reportError(
          "Only the execute command supports multiple inputs right now.");
    }
    for (StringRef inputFilename : inputAdaptorRange) {
      InputInfo info{};
      info.inputPath = std::filesystem::path{inputFilename.str()};

      std::error_code errorCode;
      info.isInputDirectory =
          std::filesystem::is_directory(info.inputPath, errorCode);
      if (errorCode) {
        return options.reportError(
            Twine("failed to check type of path: ") + inputFilename +
            " (error code: " + Twine(errorCode.message()) + ")");
      }

      if (info.isInputDirectory) {
        // If the input file is a directory, we will just assume that it is a TF
        // SavedModel. If other directory formats come up, we will need to
        // modify this logic.
        info.format = FileFormat::kTFSavedModel;
      } else {
        // The input file is not a directory, so we open and sniff the format.
        // In case the input file is MEF, it has to be aligned to the maximum
        // attribute alignment.
        info.file =
            options.openInputFileOrExit(inputFilename, maxAttributeAlignment());
        info.format = options.getInputFormat(
            info.inputPath.extension().string(), info.file->getBuffer());
      }

      inputsInfo.push_back(std::move(info));
    }
  }

  // We'll just use the final scenario's return code as the executable's
  // return code.
  int rc = EXIT_SUCCESS;

  for (auto &inputInfo : inputsInfo) {
    auto dylibOr = loadDesiredLibrary(argc, argv, *inputInfo.file,
                                      inputInfo.format, options.loadLib);
    if (failed(dylibOr))
      return options.reportError(dylibOr.getError());

    CXXDriverAPI dylib = *dylibOr;

    // If the driver command is just `-get-version` then print the version and
    // exit right away. No need to do anything else.
    if (options.command == DriverCommand::getVersion) {
      llvm::outs() << "mt: " << getModularVersionString()
                   << "\nFramework: " << dylib.getVersion() << "\n";
      return EXIT_SUCCESS;
    }

    // Setup the framework context. Despite passing argc/argv no significant
    // command line argument processing is done here.
    auto frameworkCtxOr = dylib.initFrameworkContext(argc, argv);
    if (failed(frameworkCtxOr))
      return options.reportError(frameworkCtxOr.getError());
    auto frameworkCtx = std::move(*frameworkCtxOr);

    // We'll accumulate command line arguments for the framework as we go below,
    SmallVector<std::string> dylibOptions;

    // We might want to get the framework's help string.
    if (options.command == DriverCommand::frameworkHelp) {
      if (auto err = dylib.setOptions(frameworkCtx, {"-h"}))
        return options.reportError(err.getError());
      return EXIT_SUCCESS;
    }

    // Hand over any explicitly designated unparsed options.
    dylibOptions.append(options.forceUnparsed.begin(),
                        options.forceUnparsed.end());
    dylibOptions.append(options.unparsed.begin(), options.unparsed.end());

    // Hand over save-temps and temps-dir options to framework.
    if (options.saveTemps) {
      dylibOptions.push_back("--save-temps");
      dylibOptions.push_back("--temps-dir=" + options.tempsDir.getValue());
    }

    // Tell the framework about our options.
    if (auto err = dylib.setOptions(frameworkCtx, dylibOptions))
      return options.reportError(err.getError());

    // Start processing the input in stages.  The inspect command is first,
    // because it works on any input without transformation.
    if (options.command == DriverCommand::inspect) {
      std::unique_ptr<llvm::ToolOutputFile> outputFile =
          options.getOutputFile(options.commandHasBinaryOutput());
      auto identityOr = getFileContentIdentity(
          *inputInfo.file, inputInfo.format, frameworkCtx, dylib);
      if (failed(identityOr))
        return options.reportError(identityOr.getError());

      outputFile->os() << *identityOr << ".\n";
      outputFile->keep();
      return EXIT_SUCCESS;
    }

    // If we are dealing with a directory, we need to open it.
    if (inputInfo.isInputDirectory) {
      // TODO: this should be called for all files
      auto fileOr = dylib.openPath(frameworkCtx, inputInfo.inputPath);
      if (failed(fileOr))
        return options.reportError(fileOr.getError());
      inputInfo.file = std::move(*fileOr);
    }

    // If we're stopping at compilation, emit the binary file directly to the
    // output.
    if (options.command == DriverCommand::compile) {
      auto outputFile = options.getOutputFile(options.commandHasBinaryOutput());

      // If the file is already a binary, just emit it directly.
      if (inputInfo.format == FileFormat::kMEFFile ||
          dylib.isBinary(frameworkCtx, *inputInfo.file, inputInfo.format)) {
        outputFile->os().write(inputInfo.file->getBufferStart(),
                               inputInfo.file->getBufferSize());
        outputFile->keep();
        return EXIT_SUCCESS;
      }

      // Otherwise, compile it.
      if (auto err = dylib.compileToBinary(frameworkCtx, *inputInfo.file,
                                           outputFile->os(), inputInfo.format))
        return options.reportError(err.getError());

      outputFile->keep();
      return EXIT_SUCCESS;
    }

    // If we are not inspecting, we must need a compiled format.  If this is
    // already done then we're golden, otherwise compile now.
    if (inputInfo.format != FileFormat::kMEFFile &&
        !dylib.isBinary(frameworkCtx, *inputInfo.file, inputInfo.format)) {
      if (auto err = dylib.compileToBinary(frameworkCtx, inputInfo.file,
                                           inputInfo.format))
        return options.reportError(err.getError());
    }

    // Otherwise we must be getting op-level information about the model or
    // executing the model.
    assert((options.command == DriverCommand::printMetadata ||
            options.command == DriverCommand::printConstantMetadata ||
            options.command == DriverCommand::execute) &&
           "unknown command");

    auto binaryFileOr = dylib.openBinaryFile(frameworkCtx, *inputInfo.file);
    if (failed(binaryFileOr))
      return options.reportError(binaryFileOr.getError());
    auto binaryFile = std::move(*binaryFileOr);

    // Check to see if this file contains what we expected.  If not, throw an
    // error message.
    if (auto err = dylib.checkExpectedContents(frameworkCtx, binaryFile))
      return options.reportError(err.getError());

    // If we're just getting information, then get and print it.
    if (options.command == DriverCommand::printMetadata ||
        options.command == DriverCommand::printConstantMetadata) {
      if (auto err = dylib.getMetadataFromBinary(frameworkCtx, binaryFile,
                                                 *options.getOutputFile(false),
                                                 options.command.getValue()))
        return options.reportError(err.getError());
      return EXIT_SUCCESS;
    }

    // Hand off the MEFFileHandle and the file buffer to the framework to
    // actually run the thing with all the various diagnostics.
    if (options.scenarios.empty())
      rc = dylib.runBinaryFile(frameworkCtx, std::move(inputInfo.file),
                               binaryFile);

    for (Scenario scenario : options.scenarios) {
      printf("--- (mt) Running scenario: '%s", scenario.flags.front().c_str());
      for (StringRef f : llvm::drop_begin(scenario.flags))
        printf(" %s", f.data());
      printf("' - %s \n", scenario.abortOnFailure ? "Aborting on failure."
                                                  : "Not aborting on failure.");
      fflush(stdout);

      auto fileCopy = MemoryBuffer::getMemBufferCopy(
          inputInfo.file->getBuffer(), inputInfo.file->getBufferIdentifier());

      // Add any common flags to the scenario, this will ensure that the user
      // can pass something like "result-output-style=full" to all scenarios
      // without having to specify it each time.
      scenario.flags.append(dylibOptions);

      // Reset the options.
      if (auto err = dylib.setOptions(frameworkCtx, scenario.flags))
        return options.reportError(err.getError());

      rc = dylib.runBinaryFile(frameworkCtx, std::move(fileCopy), binaryFile);
      if (rc != EXIT_SUCCESS && scenario.abortOnFailure)
        return rc;
      else if (!scenario.abortOnFailure)
        rc = EXIT_SUCCESS; // Reset `rc` for suppressed failures.
    }
  }

  return rc;
}
