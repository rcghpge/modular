//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Config/Version.h"
#include "GraphCompiler/CommonDriver/DriverInputFormat.h"
#include "GraphCompiler/CommonDriver/Utils.h"
#include "GraphCompiler/ExecutionDriver/DriverCLOptions.h"
#include "GraphCompiler/FrameworkFrontend.h"
#include "GraphCompiler/GraphCompiler/Passes.h"
#include "GraphCompiler/MEFModel/MEFModel.h"
#include "Init/Init.h"
#include "MLRT/AsyncRT/CompilerSupport/Context.h"
#include "MLRT/AsyncRT/Runtime/Utils.h"
#include "MLRT/GraphRT/MEFExecutor/MEFFile.h"
#include "Support/AssertStream.h"
#include "Support/Context.h"
#include "Support/ML/DebugPrint.h"
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

class GenericDriverOptions : public DriverCommandOptions, public DriverOptions {
public:
  //===--------------------------------------------------------------------===//
  // Sink for all unparsed options.
  //===--------------------------------------------------------------------===//

  llvm::SmallVector<std::string> forceUnparsed{};
  llvm::SmallVector<std::string> unparsed{};
};

class GenericDriverCLOptions : public DriverCommandCLOptions,
                               public DriverCLOptions {

public:
  GenericDriverOptions &options;
  GenericDriverCLOptions(int argc, char **argv, GenericDriverOptions &opts,
                         bool skipInitLLVM = false)
      : DriverCommandCLOptions(opts),
        DriverCLOptions(argc, argv, opts, skipInitLLVM), options(opts) {}

private:
  llvm::cl::OptionCategory GenericDriverCLOptionsCategory{
      "Generic driver options"};
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

static int setupMLIRAndRun(
    ContextRef context, const ExecutorDriverOptions &options,
    const char *testName, std::unique_ptr<llvm::MemoryBuffer> &&file,
    std::function<LogicalResult(CPUDevice &, MLIRContext *)> &&bodyFn) {
  if (failed(options.configureMLIRContextAndExecute(
          std::move(file), [&](MLIRContext *ctx) {
            MLRT::CPUDevice *cpuDevice = context->get<CPUDevice>();
            assert(cpuDevice && "caller should have set up cpuDevice for this");
            registerContext(*ctx, context);
            return MLRT::runWithLeakCheckedRuntime(
                testName, [&]() { return bodyFn(*cpuDevice, ctx); });
          })))
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}

static int runBinaryFile(ContextRef maxCtx, MOFContext &ctx,
                         M::BufferRef fileBuffer, const ModelList &models) {
  const ExecutorDriverOptions *driverOptions = ctx.getExecutorDriverOptions();
  assert(driverOptions && "cannot run a binary file via the API");

  auto runModels =
      [&, context = maxCtx.copy()](CPUDevice &cpuDevice,
                                   MLIRContext *mlirCtx) -> LogicalResult {
    context->emplaceIfMissing<DebugTensorPrintOptions>(
        driverOptions->resultOutputStyle, driverOptions->printPrecision,
        driverOptions->binaryOutputDir);
    bool hadFailure = false;
    for (const auto &model : models) {
      if (failed(runModel(model.get(), cpuDevice, *driverOptions))) {
        hadFailure = true;
        if (driverOptions->stopOnFirstError())
          return failure();
      }
    }
    return failure(hadFailure);
  };

  // Run it.
  std::unique_ptr<llvm::MemoryBuffer> aliasedBuffer =
      llvm::MemoryBuffer::getMemBuffer(fileBuffer->getMemBufferRef(),
                                       /*RequiresNullTerminator=*/
                                       false);

  return setupMLIRAndRun(maxCtx.copy(), *driverOptions, "Modular model",
                         std::move(aliasedBuffer), runModels);
}

int main(int argc, char **argv) {
  // Register all our command line options. Avoid calling InitLLVM as we don't
  // want to register its signal handler. The Init::Init code will register
  // our signal handler.
  GenericDriverOptions options;
  GenericDriverCLOptions parser(argc, argv, options, /*skipInitLLVM=*/true);
  llvm::cl::ParseCommandLineOptions(argc, argv, "Modular Tool (mt)\n");

  struct InputInfo {
    M::BufferRef file;
    FileFormat format = FileFormat::kUnknown;
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
      if (std::filesystem::is_directory(info.inputPath, errorCode))
        return options.reportError("unable to detect input model format");
      if (errorCode) {
        return options.reportError(
            Twine("failed to check type of path: ") + inputFilename +
            " (error code: " + Twine(errorCode.message()) + ")");
      }

      // The input file is not a directory, so we open and sniff the format.
      // In case the input file is MEF, it has to be aligned to the maximum
      // attribute alignment.
      std::unique_ptr<llvm::MemoryBuffer> inputFile =
          options.openInputFileOrExit(inputFilename, maxAttributeAlignment());
      info.file = Buffer::take(std::move(inputFile));
      info.format = options.getInputFormat(info.inputPath.extension().string(),
                                           info.file->getBuffer());

      inputsInfo.push_back(std::move(info));
    }
  }

  // We'll just use the final scenario's return code as the executable's
  // return code.
  int rc = EXIT_SUCCESS;

  // Create a context used by this command (compile/execute/benchmark) run.
  Init::Options contextOptions;
  contextOptions.withCPUDeviceOptions(options);
  auto contextOr =
      Init::createContext(options.getProgramName(), contextOptions);
  if (contextOr.isError())
    return options.reportError(contextOr.getError());

  // If the driver command is just `-get-version` then print the version and
  // exit right away. No need to do anything else.
  if (options.command == DriverCommand::getVersion) {
    // llvm::formatv does not support octal, so do it the hard way.
    char profilingLevelStr[16];
    snprintf(profilingLevelStr, sizeof(profilingLevelStr), "%#o",
             MODULAR_ASYNCRT_MAX_PROFILING_LEVEL);
    llvm::outs() << "mt: " << getModularVersionString()
                 << "\nmt profiling level: " << profilingLevelStr << "\n";
    return EXIT_SUCCESS;
  }

  // Setup the framework context.
  MOFContext frameworkCtx(contextOr->copy(), argc, argv);

  // We might want to get the framework's help string.
  if (options.command == DriverCommand::frameworkHelp) {
    if (auto err = frameworkCtx.setOptions({"-h"}))
      return options.reportError(err.getError());
    return EXIT_SUCCESS;
  }

  // We'll accumulate command line arguments for the framework as we go below,
  SmallVector<std::string> dylibOptions;

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
  if (auto err = frameworkCtx.setOptions(dylibOptions))
    return options.reportError(err.getError());
  frameworkCtx.setDriverOptions(options);

  for (auto &inputInfo : inputsInfo) {

    // Start processing the input in stages.  The inspect command is first,
    // because it works on any input without transformation.
    if (options.command == DriverCommand::inspect) {
      std::unique_ptr<llvm::ToolOutputFile> outputFile =
          options.getOutputFile(options.commandHasBinaryOutput());
      auto identityOr =
          getFileContentIdentity(inputInfo.file.copy(), inputInfo.format);
      if (failed(identityOr))
        return options.reportError(identityOr.getError());

      outputFile->os() << *identityOr << ".\n";
      outputFile->keep();
      return EXIT_SUCCESS;
    }

    // If we're stopping at compilation, emit the binary file directly to the
    // output.
    if (options.command == DriverCommand::compile) {
      auto outputFile = options.getOutputFile(options.commandHasBinaryOutput());

      // If the file is already a binary, just emit it directly.
      if (inputInfo.format == FileFormat::kMEFFile) {
        outputFile->os().write(inputInfo.file->getBufferStart(),
                               inputInfo.file->getBufferSize());
        outputFile->keep();
        return EXIT_SUCCESS;
      }

      // Otherwise, compile it.
      auto bufferOr =
          frameworkCtx.compileToBinary(inputInfo.file.copy(), inputInfo.format);
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
    if (inputInfo.format != FileFormat::kMEFFile) {
      auto bufferOr =
          frameworkCtx.compileToBinary(inputInfo.file.copy(), inputInfo.format);
      if (bufferOr.isError())
        return options.reportError(bufferOr.getError());

      inputInfo.file = bufferOr.takeValue();
    }

    // Otherwise we must be getting op-level information about the model or
    // executing the model.
    if (!(options.command == DriverCommand::printMetadata ||
          options.command == DriverCommand::printConstantMetadata ||
          options.command == DriverCommand::execute)) {
      return options.reportError("Unknown command:");
    }

    auto mefFileOr = MEFFile::open(inputInfo.file->getBuffer());
    if (failed(mefFileOr))
      return options.reportError(mefFileOr.getError());
    auto modelsOr = M::loadMGPMEFFile(*mefFileOr);
    if (modelsOr.isError())
      return options.reportError(modelsOr.getError());

    // Hand off the MEFFileHandle and the file buffer to the framework to
    // actually run the thing with all the various diagnostics.
    rc = runBinaryFile(contextOr->copy(), frameworkCtx, inputInfo.file.copy(),
                       *modelsOr);
  }

  return rc;
}
