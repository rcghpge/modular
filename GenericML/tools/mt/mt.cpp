//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/Utils.h"
#include "Config/Version.h"
#include "GenericML/CommonDriver/DriverInputFormat.h"
#include "GenericML/CommonDriver/Utils.h"
#include "GenericML/ExecutionDriver/DriverCLOptions.h"
#include "GenericML/FrameworkFrontend.h"
#include "GenericML/GraphCompiler/Passes.h"
#include "GenericML/MEFModel/MEFModel.h"
#include "GenericML/MOFOptions.h"
#include "GraphRT/MEFExecutor/MEFFile.h"
#include "Init/Init.h"
#include "Support/AssertStream.h"
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
                             public AsyncRT::RuntimeOptions {
public:
  //===--------------------------------------------------------------------===//
  // Sink for all unparsed options.
  //===--------------------------------------------------------------------===//

  llvm::SmallVector<std::string> forceUnparsed{};
  llvm::SmallVector<std::string> unparsed{};
};

class GenericDriverCLOptions : public DriverCommandCLOptions,
                               public DriverInputFormatCLOptions,
                               public CommonCLOptions,
                               public AsyncRT::RuntimeCLOptions {

public:
  GenericDriverOptions &options;
  GenericDriverCLOptions(int argc, char **argv, GenericDriverOptions &opts,
                         bool skipInitLLVM = false)
      : DriverCommandCLOptions(opts), DriverInputFormatCLOptions(opts),
        CommonCLOptions(argc, argv, opts, skipInitLLVM),
        AsyncRT::RuntimeCLOptions(opts), options(opts) {}

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
    std::function<LogicalResult(Runtime &, MLIRContext *)> &&bodyFn) {
  if (failed(options.configureMLIRContextAndExecute(
          std::move(file), [&](MLIRContext *ctx) {
            AsyncRT::Runtime *runtime = context->get<Runtime>();
            assert(runtime && "caller should have set up runtime for this");
            registerContext(*ctx, context);
            return AsyncRT::runWithLeakCheckedRuntime(
                testName, [&]() { return bodyFn(*runtime, ctx); });
          })))
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}

static int runBinaryFile(ContextRef maxCtx, MOFContext &ctx,
                         M::BufferRef fileBuffer, ModelCollection &models) {
  MOF::MOFLoweringOptions mofLoweringOptions;
  MO::MoToMgpOptions moToMgpOptions;
  ctx.overrideLoweringOptions(mofLoweringOptions, moToMgpOptions);
  const ExecutorDriverOptions *driverOptions = ctx.getExecutorDriverOptions();
  assert(driverOptions && "cannot run a binary file via the API");

  auto runModels = [&, context = maxCtx.copy()](
                       Runtime &runtime, MLIRContext *ctx) -> LogicalResult {
    mofLoweringOptions.setupContext(context);
    for (const auto &model : models) {
      if (failed(runModel(model.get(), runtime, *driverOptions))) {
        if (driverOptions->stopOnFirstError())
          return failure();
      }
    }
    return success();
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
  contextOptions.withRuntimeOptions(options);
  auto contextOr =
      Init::createContext(options.getProgramName(), contextOptions);
  if (contextOr.isError())
    return options.reportError(contextOr.getError());

  for (auto &inputInfo : inputsInfo) {
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

    // We'll accumulate command line arguments for the framework as we go below,
    SmallVector<std::string> dylibOptions;

    // We might want to get the framework's help string.
    if (options.command == DriverCommand::frameworkHelp) {
      if (auto err = frameworkCtx.setOptions({"-h"}))
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
    if (auto err = frameworkCtx.setOptions(dylibOptions))
      return options.reportError(err.getError());

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
    auto modelsOr = MO::loadMGPMEFFile(*mefFileOr);
    if (modelsOr.isError())
      return options.reportError(modelsOr.getError());

    // Hand off the MEFFileHandle and the file buffer to the framework to
    // actually run the thing with all the various diagnostics.
    rc = runBinaryFile(contextOr->copy(), frameworkCtx, inputInfo.file.copy(),
                       **modelsOr);
  }

  return rc;
}
