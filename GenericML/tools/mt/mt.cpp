//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Config/Version.h"
#include "GenericML/Driver/CXXDriverAPI.h"
#include "GenericML/Driver/DriverAPI.h"
#include "GenericML/Driver/DriverInputFormat.h"
#include "Support/AssertStream.h"
#include "Support/Init/Init.h"
#include "Support/ML/SizeUtils.h"
#include "Support/Process.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ToolOutputFile.h"

#include <filesystem>

using namespace M;
using llvm::MemoryBuffer;

namespace {

//===--------------------------------------------------------------------===//
// GenericDriverCLOptions
//===--------------------------------------------------------------------===//

class GenericDriverOptions : public DriverCommandOptions,
                             public DriverInputFormatOptions,
                             public CommonOptions,
                             public LLCL::RuntimeOptions {
public:
  //===--------------------------------------------------------------------===//
  // Force loading a specific dylib
  //===--------------------------------------------------------------------===//

  std::string loadLib{""};
  //===--------------------------------------------------------------------===//
  // Sink for all unparsed options.
  //===--------------------------------------------------------------------===//

  llvm::SmallVector<std::string> forceUnparsed{};
  llvm::SmallVector<std::string> unparsed{};
};

class GenericDriverCLOptions : public DriverCommandCLOptions,
                               public DriverInputFormatCLOptions,
                               public CommonCLOptions,
                               public LLCL::RuntimeCLOptions {

public:
  GenericDriverOptions &options;
  GenericDriverCLOptions(int argc, char **argv, GenericDriverOptions &opts,
                         bool skipInitLLVM = false)
      : DriverCommandCLOptions(opts), DriverInputFormatCLOptions(opts),
        CommonCLOptions(argc, argv, opts, skipInitLLVM),
        LLCL::RuntimeCLOptions(opts), options(opts) {}

private:
  llvm::cl::OptionCategory GenericDriverCLOptionsCategory{
      "Generic driver options"};
  //===--------------------------------------------------------------------===//
  // Force loading a specific dylib
  //===--------------------------------------------------------------------===//

  M::cl::MOpt<std::string, true> loadLib{
      "load", cl::desc("Load the specified framework directly."),
      llvm::cl::cat(GenericDriverCLOptionsCategory),
      llvm::cl::location(options.loadLib)};

  //===--------------------------------------------------------------------===//
  // Sink for all unparsed options.
  //===--------------------------------------------------------------------===//

  M::cl::MListOpt<std::string, llvm::SmallVector<std::string>> forceUnparsed{
      "X", cl::desc("Arguments to be passed to the framework."),
      llvm::cl::cat(GenericDriverCLOptionsCategory),
      llvm::cl::location(options.forceUnparsed)};

  M::cl::MListOpt<std::string, llvm::SmallVector<std::string>> unparsed{
      llvm::cl::Sink,
      cl::desc("Arguments to be passed directly to the framework."),
      llvm::cl::cat(GenericDriverCLOptionsCategory),
      llvm::cl::location(options.unparsed)};
};
} // namespace

// TODO(#1639): Remove.
/// Returns a string that describes the contents of the file. Given that this
/// driver cannot link the MEF libraries, it relies on the symbols provided by
/// the framework supplied with dylib if the input is a MEF file.
static M::ErrorOr<std::string>
getFileContentIdentity(M::BufferRef file, FileFormat format,
                       FrameworkContextHandle ctxHandle, CXXDriverAPI dylib) {
  switch (format) {
  default:
    return "input has unknown contents";
  case FileFormat::kTFDialect:
    return "input is a TensorFlow model in MLIR form";
  case FileFormat::kTFSavedModel:
    return "input is a TensorFlow model in SavedModel form";
  case FileFormat::kTFPDialect:
    return "input is a compiled TensorFlow MEF model in MLIR form";
  case FileFormat::kFauxDialect:
    return "input is Faux model in MLIR form";
  case FileFormat::kMODialect:
    [[fallthrough]];
  case FileFormat::kMGPDialect:
    return "input is a Modular model";
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
  case FileFormat::kTorchPDialect:
    return "input is a PyTorch model in MLIR form consisting of primitive ops";
  case FileFormat::kMEFFile: {
    auto fileHandleOr = dylib.openBinaryFile(ctxHandle, file.copy());
    if (fileHandleOr.isError())
      return fileHandleOr.takeError();
    return dylib.sniffBinaryFileContents(ctxHandle, std::move(*fileHandleOr));
  }
  }
}

int main(int argc, char **argv) {
  // Register all our command line options.  We expect that the subframeworks
  // will not call InitLLVM, and that is the responsibility of the driver.
  GenericDriverOptions options;
  GenericDriverCLOptions parser(argc, argv, options);

  llvm::cl::ParseCommandLineOptions(argc, argv, "Modular Tool (mt)\n");

  if (options.disableTFOneDNN) {
    // TODO(#20024): Disable TF's oneDNN replacement kernels. Better solution?
    (void)setProcessEnv("TF_ENABLE_ONEDNN_OPTS", "0");
  }

  struct InputInfo {
    M::BufferRef file;
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
        if (isTFSavedModelDirectory(info.inputPath))
          info.format = FileFormat::kTFSavedModel;
        else
          return options.reportError("unable to detect input model format");
      } else {
        // The input file is not a directory, so we open and sniff the format.
        // In case the input file is MEF, it has to be aligned to the maximum
        // attribute alignment.
        std::unique_ptr<llvm::MemoryBuffer> inputFile =
            options.openInputFileOrExit(inputFilename, maxAttributeAlignment());
        info.file = Buffer::take(std::move(inputFile));
        info.format = options.getInputFormat(
            info.inputPath.extension().string(), info.file->getBuffer());
      }

      inputsInfo.push_back(std::move(info));
    }
  }

  // We'll just use the final scenario's return code as the executable's
  // return code.
  int rc = EXIT_SUCCESS;

  // Create a context used by this command (compile/execute/benchmark) run.
  Init::Options contextOptions;
  contextOptions.withRuntimeOptions(options);
  auto contextOr =
      Init::createContext(options.getProgramName(), contextOptions);
  if (contextOr.isError())
    return options.reportError(contextOr.getError());

  for (auto &inputInfo : inputsInfo) {
    auto dylibOr = loadDesiredLibrary(argc, argv, inputInfo.file.copy(),
                                      inputInfo.format, options.loadLib);
    if (failed(dylibOr))
      return options.reportError(dylibOr.getError());

    CXXDriverAPI dylib = *dylibOr;

    // If the driver command is just `-get-version` then print the version and
    // exit right away. No need to do anything else.
    if (options.command == DriverCommand::getVersion) {
      // llvm::formatv does not support octal, so do it the hard way.
      char profilingLevelStr[16];
      snprintf(profilingLevelStr, sizeof(profilingLevelStr), "%#o",
               MODULAR_LLCL_MAX_PROFILING_LEVEL);
      llvm::outs() << "mt: " << getModularVersionString()
                   << "\nFramework: " << dylib.getVersion()
                   << "\nmt profiling level: " << profilingLevelStr << "\n";
      return EXIT_SUCCESS;
    }

    // Setup the framework context.
    auto frameworkCtxOr =
        dylib.initFrameworkContext(contextOr->copy(), argc, argv);
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
      dylibOptions.push_back("--temps-dir=" + options.tempsDir);
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
          inputInfo.file.copy(), inputInfo.format, frameworkCtx, dylib);
      if (failed(identityOr))
        return options.reportError(identityOr.getError());

      outputFile->os() << *identityOr << ".\n";
      outputFile->keep();
      return EXIT_SUCCESS;
    }

    // If we are dealing with a directory, we need to open it.
    if (inputInfo.isInputDirectory) {
      // TODO: this should be called for all files
      auto bufferOr = dylib.openPath(frameworkCtx, inputInfo.inputPath);
      if (bufferOr.isError())
        return options.reportError(bufferOr.getError());
      inputInfo.file = bufferOr.takeValue();
    }

    // If we're stopping at compilation, emit the binary file directly to the
    // output.
    if (options.command == DriverCommand::compile) {
      auto outputFile = options.getOutputFile(options.commandHasBinaryOutput());

      // If the file is already a binary, just emit it directly.
      if (inputInfo.format == FileFormat::kMEFFile ||
          dylib.isBinary(frameworkCtx, inputInfo.file.copy(),
                         inputInfo.format)) {
        outputFile->os().write(inputInfo.file->getBufferStart(),
                               inputInfo.file->getBufferSize());
        outputFile->keep();
        return EXIT_SUCCESS;
      }

      // Otherwise, compile it.
      auto bufferOr = dylib.compileToBinary(frameworkCtx, inputInfo.file.copy(),
                                            inputInfo.format);
      if (bufferOr.isError())
        return options.reportError(bufferOr.getError());

      M::BufferRef buffer = bufferOr.takeValue();
      outputFile->os().write(buffer->getBufferStart(), buffer->getBufferSize());
      outputFile->keep();
      return EXIT_SUCCESS;
    }

    M::BufferRef buffer;

    // If we are not inspecting, we must need a compiled format.  If this is
    // already done then we're golden, otherwise compile now.
    if (inputInfo.format != FileFormat::kMEFFile &&
        !dylib.isBinary(frameworkCtx, inputInfo.file.copy(),
                        inputInfo.format)) {
      auto bufferOr = dylib.compileToBinary(frameworkCtx, inputInfo.file.copy(),
                                            inputInfo.format);
      if (bufferOr.isError())
        return options.reportError(bufferOr.getError());

      inputInfo.file = bufferOr.takeValue();
    }

    // Otherwise we must be getting op-level information about the model or
    // executing the model.
    ASSERT_STREAM((options.command == DriverCommand::printMetadata ||
                   options.command == DriverCommand::printConstantMetadata ||
                   options.command == DriverCommand::execute),
                  << "unknown command ");

    auto binaryFileOr =
        dylib.openBinaryFile(frameworkCtx, inputInfo.file.copy());
    if (failed(binaryFileOr))
      return options.reportError(binaryFileOr.getError());
    auto binaryFile = std::move(*binaryFileOr);

    // Check to see if this file contains what we expected.  If not, throw an
    // error message.
    if (auto err = dylib.checkExpectedContents(frameworkCtx, binaryFile))
      return options.reportError(err.getError());

    // Hand off the MEFFileHandle and the file buffer to the framework to
    // actually run the thing with all the various diagnostics.
    rc = dylib.runBinaryFile(frameworkCtx, inputInfo.file.copy(), binaryFile);
  }

  return rc;
}
