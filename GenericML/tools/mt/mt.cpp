//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#include "Config/Config.h"
#include "GenericML/Driver/CXXDriverAPI.h"
#include "GenericML/Driver/DriverAPI.h"
#include "GenericML/Driver/DriverInputFormat.h"
#include "GenericML/Support/ModelCollection.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "llvm/Support/ErrorHandling.h"
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
/// cannot link the BEF libraries, we have to rely on the symbols provided by
/// the framework if the input is a BEF file.
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
    return "input is a compiled TFLite BEF model in MLIR form";
  case FileFormat::kTFDialect:
    return "input is a TensorFlow model in MLIR form";
  case FileFormat::kTFSavedModel:
    return "input is a TensorFlow model in SavedModel form";
  case FileFormat::kTFPDialect:
    return "input is a compiled TensorFlow BEF model in MLIR form";
  case FileFormat::kMOPDialect:
    return "input is a compiled Modular BEF model in MLIR form";
  case FileFormat::kFauxDialect:
    return "input is Faux model in MLIR form";
  case FileFormat::kBEFFile: {
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

  FileFormat format = FileFormat::kUnknown;

  // If we don't have a regular input file (and we're attempting to run
  // something that doesn't take any input), then we have to pass in an empty
  // memory buffer which will result in an unknown file format. We want to
  // preserve the stdin behavior for cases where the command does take input
  // because we might be piping something into the tool.
  std::unique_ptr<MemoryBuffer> file = MemoryBuffer::getMemBuffer("", "empty");
  bool isInputDirectory = false;
  std::filesystem::path inputPath;
  if (options.commandTakesInput() || options.inputFilename != "-") {
    std::error_code errorCode;
    inputPath = std::filesystem::path{options.inputFilename.getValue()};
    isInputDirectory = std::filesystem::is_directory(inputPath, errorCode);
    if (errorCode) {
      return options.reportError(
          Twine("failed to check type of path: ") + inputPath.string() +
          " (error code: " + Twine(errorCode.message()) + ")");
    }

    if (isInputDirectory) {
      // If the input file is a directory, we will just assume that it is a TF
      // SavedModel. If other directory formats come up, we will need to modify
      // this logic.
      format = FileFormat::kTFSavedModel;
    } else {
      // The input file is not a directory, so we open and sniff the format.
      file = options.openInputFileOrExit();
      format = options.getInputFormat(file->getBuffer());
    }
  }

  auto dylibOr = loadDesiredLibrary(argc, argv, *file, format, options.loadLib);
  if (failed(dylibOr))
    return options.reportError(dylibOr.getError());

  CXXDriverAPI dylib = *dylibOr;

  // If the driver command is just `-get-version` then print the version and
  // exit right away. No need to do anything else.
  if (options.command == DriverCommand::getVersion) {
    llvm::outs() << "mt: " << MODULAR_VERSION_STRING
                 << "\nFramework: " << dylib.getVersion() << "\n";
    return 0;
  }

  // Setup the framework context.
  auto frameworkCtxOr = dylib.initFrameworkContext(argc, argv);
  if (failed(frameworkCtxOr))
    return options.reportError(frameworkCtxOr.getError());
  auto frameworkCtx = std::move(*frameworkCtxOr);

  // We might want to get the framework's help string.
  if (options.command == DriverCommand::frameworkHelp) {
    if (auto err = dylib.setOptions(frameworkCtx, {"-h"}))
      return options.reportError(err.getError());

    return 0;
  }

  // Hand over any explicitly designated unparsed options.
  SmallVector<std::string> unparsed(options.forceUnparsed.begin(),
                                    options.forceUnparsed.end());
  unparsed.append(options.unparsed.begin(), options.unparsed.end());
  if (auto err = dylib.setOptions(frameworkCtx, unparsed))
    return options.reportError(err.getError());

  // Hand over save-temps and temps-dir options to framework.
  if (options.saveTemps) {
    std::string tempsDirOpt = "--temps-dir=" + options.tempsDir.getValue();
    if (auto err =
            dylib.setOptions(frameworkCtx, {"--save-temps", tempsDirOpt}))
      return options.reportError(err.getError());
  }

  // Start processing the input in stages.  The inspect command is first,
  // because it works on any input without transformation.
  if (options.command == DriverCommand::inspect) {
    std::unique_ptr<llvm::ToolOutputFile> outputFile =
        options.getOutputFile(options.commandHasBinaryOutput());
    auto identityOr =
        getFileContentIdentity(*file, format, frameworkCtx, dylib);
    if (identityOr.isError()) {
      options.reportError(identityOr.getError());
      return 1;
    }

    outputFile->os() << *identityOr << ".\n";
    outputFile->keep();
    return 0;
  }

  // If we are dealing with a directory, we need to open it.
  if (isInputDirectory) {
    // TODO: this should be called for all files
    auto fileOr = dylib.openPath(frameworkCtx, inputPath);
    if (failed(fileOr))
      return options.reportError(fileOr.getError());
    file = std::move(*fileOr);
  }

  // If we're stopping at compilation, emit the binary file directly to the
  // output.
  if (options.command == DriverCommand::compile) {
    auto outputFile = options.getOutputFile(options.commandHasBinaryOutput());

    // If the file is already a binary, just emit it directly.
    if (format == FileFormat::kBEFFile ||
        dylib.isBinary(frameworkCtx, *file, format)) {
      outputFile->os().write(file->getBufferStart(), file->getBufferSize());
      outputFile->keep();
      return 0;
    }

    // Otherwise, compile it.
    if (failed(dylib.compileToBinary(frameworkCtx, *file, outputFile->os(),
                                     format)))
      return 1;

    outputFile->keep();
    return 0;
  }

  // If we are not inspecting, we must need a compiled format.  If this is
  // already done then we're golden, otherwise compile now.
  if (format != FileFormat::kBEFFile &&
      !dylib.isBinary(frameworkCtx, *file, format)) {
    if (failed(dylib.compileToBinary(frameworkCtx, file, format)))
      return 1;
  }

  // Otherwise we must be getting op-level information about the model or
  // executing the model.
  assert((options.command == DriverCommand::printMetadata ||
          options.command == DriverCommand::printConstantMetadata ||
          options.command == DriverCommand::execute) &&
         "unknown command");

  auto binaryFileOr = dylib.openBinaryFile(frameworkCtx, *file);
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
    return 0;
  }

  // Hand off the BEFFileHandle and the file buffer to the framework to actually
  // run the thing with all the various diagnostics.
  if (options.scenarios.empty())
    return dylib.runBinaryFile(frameworkCtx, std::move(file), binaryFile);

  // We'll just use the final scenario's return code as the executable's return
  // code.
  int rc = EXIT_SUCCESS;
  for (Scenario scenario : options.scenarios) {
    printf("--- (mt) Running scenario: '%s", scenario.flags.front().c_str());
    for (StringRef f : llvm::drop_begin(scenario.flags))
      printf(" %s", f.data());
    printf("' - %s \n", scenario.abortOnFailure ? "Aborting on failure."
                                                : "Not aborting on failure.");
    fflush(stdout);

    auto fileCopy = MemoryBuffer::getMemBufferCopy(file->getBuffer(),
                                                   file->getBufferIdentifier());

    // Add any common flags to the scenario, this will ensure that the user can
    // pass something like "result-output-style=full" to all scenarios without
    // having to specify it each time.
    scenario.flags.append(unparsed);

    // Set the options.
    if (auto err = dylib.setOptions(frameworkCtx, scenario.flags))
      return options.reportError(err.getError());

    rc = dylib.runBinaryFile(frameworkCtx, std::move(fileCopy), binaryFile);
    if (rc != EXIT_SUCCESS && scenario.abortOnFailure)
      return rc;
    else if (!scenario.abortOnFailure)
      rc = EXIT_SUCCESS; // Reset `rc` for suppressed failures.
  }

  return rc;
}
