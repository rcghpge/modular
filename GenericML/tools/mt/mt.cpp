//===- mt.cpp -------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/DriverAPI.h"
#include "GenericML/Driver/DriverCLOptions.h"
#include "GenericML/Support/ModelCollection.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "PlatformDylib.h.inc"
#include "Support/VCSRevision.h"
#include "llvm/Support/DynamicLibrary.h"
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

//===--------------------------------------------------------------------===//
// RAII-style managed handles (Framework & Binary)
//===--------------------------------------------------------------------===//

/// This class provides a managed FrameworkContextHandle that we can use in `mt`
/// to automatically free the given framework handle.
class ManagedFrameworkHandle {
public:
  ManagedFrameworkHandle(decltype(&M_freeFrameworkContext) deleter)
      : deleter(deleter) {}
  ManagedFrameworkHandle(ManagedFrameworkHandle &&other) {
    std::swap(deleter, other.deleter);
    std::swap(handle, other.handle);
  }
  ~ManagedFrameworkHandle() {
    if (deleter)
      deleter(handle);
  }

  FrameworkContextHandle *get() { return &handle; }
  operator FrameworkContextHandle() { return handle; }

  decltype(&M_freeFrameworkContext) deleter = nullptr;
  FrameworkContextHandle handle = {};

private:
  // Make this non-copyable.
  ManagedFrameworkHandle(const ManagedFrameworkHandle &other) = delete;
};

/// This class provides a managed ModelBinaryHandle that we can use in `mt` to
/// automatically close the given binary file.
class ManagedBinaryHandle {
public:
  ManagedBinaryHandle(decltype(&M_closeBinaryFile) deleter)
      : deleter(deleter), handle(ModelBinaryHandle{nullptr}) {}
  ManagedBinaryHandle(ManagedBinaryHandle &&other) {
    std::swap(deleter, other.deleter);
    std::swap(handle, other.handle);
  }
  ~ManagedBinaryHandle() {
    if (deleter)
      deleter(handle);
  }

  ModelBinaryHandle *get() { return &handle; }
  operator ModelBinaryHandle() { return handle; }

  decltype(&M_closeBinaryFile) deleter = nullptr;
  ModelBinaryHandle handle = {};

private:
  // Make this non-copyable.
  ManagedBinaryHandle(const ManagedBinaryHandle &other) = delete;
};

//===--------------------------------------------------------------------===//
// Driver API helpers
//===--------------------------------------------------------------------===//
class DriverAPI {
public:
  static M::ErrorOr<DriverAPI> getFromDylib(llvm::sys::DynamicLibrary dylib);

  std::string getVersion() const {
    std::string rev;
    getVersionStringHandle(&rev);
    return rev;
  }

  ErrorOr<ManagedFrameworkHandle> initFrameworkContext(int &argc,
                                                       char **&argv) {
    ManagedFrameworkHandle handle(freeFrameworkContextHandle);
    RETURN_ERROR(initFrameworkContextHandle(handle.get(), argc, argv));
    return std::move(handle);
  }

  ErrorOrSuccess setOptions(FrameworkContextHandle ctx,
                            llvm::ArrayRef<std::string> opts) {
    return setOptionsHandle(ctx, opts);
  }

  ErrorOrSuccess compileToBinary(FrameworkContextHandle ctx,
                                 std::unique_ptr<llvm::MemoryBuffer> &file,
                                 FileFormat &format) {
    return compileToBinaryHandle(ctx, file, format);
  }

  ErrorOrSuccess getMetadataFromBinary(FrameworkContextHandle ctx,
                                       ModelBinaryHandle handle,
                                       llvm::ToolOutputFile &outputFile,
                                       DriverCommand command) {
    return getMetadataFromBinaryHandle(ctx, handle, outputFile, (int)command);
  }

  M::ErrorOr<ManagedBinaryHandle>
  openBinaryFile(FrameworkContextHandle ctx, const llvm::MemoryBuffer &file) {
    ManagedBinaryHandle handle(closeBinaryFileHandle);
    RETURN_ERROR(openBinaryFileHandle(ctx, file, handle.get()));
    return std::move(handle);
  }

  ErrorOrSuccess checkExpectedContents(FrameworkContextHandle ctx,
                                       ModelBinaryHandle handle) {
    return checkExpectedContentsHandle(ctx, handle);
  }

  M::ErrorOr<std::string> sniffBinaryFileContents(M::FrameworkContextHandle ctx,
                                                  M::ModelBinaryHandle handle) {
    std::string out;
    sniffBinaryFileContentsHandle(ctx, handle, &out);
    return out;
  }

  int runBinaryFile(FrameworkContextHandle ctx,
                    std::unique_ptr<llvm::MemoryBuffer> &&fileBuffer,
                    ModelBinaryHandle handle) {
    return runBinaryFileHandle(ctx, std::move(fileBuffer), handle);
  }

private:
  DriverAPI(llvm::sys::DynamicLibrary dylib) : dylib(dylib) {}

  template <typename SymT>
  ErrorOrSuccess fallibleGetSymbol(StringRef symName, SymT *symbol);

  llvm::sys::DynamicLibrary dylib;

  decltype(&M_getVersionString) getVersionStringHandle;
  decltype(&M_initFrameworkContext) initFrameworkContextHandle;
  decltype(&M_freeFrameworkContext) freeFrameworkContextHandle;
  decltype(&M_setOptions) setOptionsHandle;
  decltype(&M_compileToBinary) compileToBinaryHandle;
  decltype(&M_getMetadataFromBinary) getMetadataFromBinaryHandle;
  decltype(&M_checkExpectedContents) checkExpectedContentsHandle;
  decltype(&M_openBinaryFile) openBinaryFileHandle;
  decltype(&M_closeBinaryFile) closeBinaryFileHandle;
  decltype(&M_sniffBinaryFileContents) sniffBinaryFileContentsHandle;
  decltype(&M_runBinaryFile) runBinaryFileHandle;
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

//===--------------------------------------------------------------------===//
// DriverAPI class implementation
//===--------------------------------------------------------------------===//

template <typename SymT>
ErrorOrSuccess DriverAPI::fallibleGetSymbol(StringRef symName, SymT *symbol) {
  auto sym = dylib.getAddressOfSymbol(symName.data());
  if (!sym)
    return Error("could not load symbol: '" + symName + "'");

  *symbol = (SymT)sym;
  return success();
}

M::ErrorOr<DriverAPI> DriverAPI::getFromDylib(llvm::sys::DynamicLibrary dylib) {
  DriverAPI api(dylib);

  RETURN_ERROR(
      api.fallibleGetSymbol("M_getVersionString", &api.getVersionStringHandle));
  RETURN_ERROR(api.fallibleGetSymbol("M_initFrameworkContext",
                                     &api.initFrameworkContextHandle));
  RETURN_ERROR(api.fallibleGetSymbol("M_freeFrameworkContext",
                                     &api.freeFrameworkContextHandle));
  RETURN_ERROR(api.fallibleGetSymbol("M_setOptions", &api.setOptionsHandle));
  RETURN_ERROR(
      api.fallibleGetSymbol("M_compileToBinary", &api.compileToBinaryHandle));
  RETURN_ERROR(api.fallibleGetSymbol("M_getMetadataFromBinary",
                                     &api.getMetadataFromBinaryHandle));
  RETURN_ERROR(api.fallibleGetSymbol("M_checkExpectedContents",
                                     &api.checkExpectedContentsHandle));
  RETURN_ERROR(
      api.fallibleGetSymbol("M_openBinaryFile", &api.openBinaryFileHandle));
  RETURN_ERROR(
      api.fallibleGetSymbol("M_closeBinaryFile", &api.closeBinaryFileHandle));
  RETURN_ERROR(api.fallibleGetSymbol("M_sniffBinaryFileContents",
                                     &api.sniffBinaryFileContentsHandle));
  RETURN_ERROR(
      api.fallibleGetSymbol("M_runBinaryFile", &api.runBinaryFileHandle));

  return api;
}

static constexpr char tflLibraryName[] = SYSTEM_SHARED_LIBRARY("tfl");

/// Get a string that describes the contents of the file. Given that this driver
/// cannot link the BEF libraries, we have to rely on the symbols provided by
/// the framework if the input is a BEF file.
static M::ErrorOr<std::string>
getFileContentIdentity(const llvm::MemoryBuffer &file, FileFormat format,
                       FrameworkContextHandle ctxHandle, DriverAPI dylib) {
  switch (format) {
  default:
    return "input has unknown contents";
  case FileFormat::kTFLFlatBuffer:
    return "input is a binary TFLite flatbuffer file";
  case FileFormat::kTFLFlatBufferJSON:
    return "input is a TFLite flatbuffer file in JSON format";
  case FileFormat::kTFLDialect:
    return "input is TFLite model in MLIR form";
  case FileFormat::kTFLPDialect:
    return "input is a compiled TFLite BEF model in MLIR form";
  case FileFormat::kFauxDialect:
    return "input is Faux model in MLIR form";
  case FileFormat::kBEFFile: {
    UNWRAP_ERROR(fileHandle, dylib.openBinaryFile(ctxHandle, file));
    return dylib.sniffBinaryFileContents(ctxHandle, fileHandle);
  }
  }
}

static M::ErrorOr<DriverAPI> loadDesiredLibrary(int &argc, char **&argv,
                                                const llvm::MemoryBuffer &file,
                                                FileFormat inputFormat,
                                                StringRef forceLoad) {
  std::string errorMsg;

  // Wrap getPermanentLibrary with a thing that extracts the error message and
  // returns it as an Error.
  auto fallibleLibraryLoad = [&](StringRef name) -> M::ErrorOr<DriverAPI> {
    auto dylib =
        llvm::sys::DynamicLibrary::getPermanentLibrary(name.data(), &errorMsg);
    if (!dylib.isValid())
      return Error("failed to load library " + name + ": " + errorMsg);
    return DriverAPI::getFromDylib(dylib);
  };

  // If we have been forced to load a specific lib, just load it and be done.
  if (!forceLoad.empty()) {
    auto path = std::filesystem::path(forceLoad.str());
    // If there's no extension, then prefix and suffix with system shared
    // library prefix/suffix.
    if (!path.has_extension()) {
      path = path.replace_filename(SYSTEM_SHARED_LIBRARY_PREFIX +
                                   path.filename().string())
                 .replace_extension(SYSTEM_SHARED_LIBRARY_SUFFIX);
    }

    return fallibleLibraryLoad(path.string());
  }

  // Switch over the input format. The BEF case is a little tricky because we
  // have to load *a* framework in order to actually sniff the contents of the
  // BEF file.
  switch (inputFormat) {
  case FileFormat::kUnknown:
    return Error("unknown file format and no framework library specified.");
  case FileFormat::kTFLFlatBuffer:
    LLVM_FALLTHROUGH;
  case FileFormat::kTFLFlatBufferJSON:
    LLVM_FALLTHROUGH;
  case FileFormat::kTFLDialect:
    LLVM_FALLTHROUGH;
  case FileFormat::kTFLPDialect:
    return fallibleLibraryLoad(tflLibraryName);
  case FileFormat::kFauxDialect:
    return Error("Faux is not yet supported.");
  case FileFormat::kBEFFile: {
    auto tflDylibOr = fallibleLibraryLoad(tflLibraryName);
    if (failed(tflDylibOr))
      return tflDylibOr.takeError();
    auto tflDylib = *tflDylibOr;

    UNWRAP_ERROR(frameworkCtx, tflDylib.initFrameworkContext(argc, argv));
    UNWRAP_ERROR(binaryHandle, tflDylib.openBinaryFile(frameworkCtx, file));
    UNWRAP_ERROR(contents,
                 tflDylib.sniffBinaryFileContents(frameworkCtx, binaryHandle));

    if (StringRef(contents).contains_insensitive("tflite"))
      return tflDylib;

    if (StringRef(contents).contains_insensitive("faux"))
      return Error("Faux is not yet supported.");

    return Error("unknown BEF file contents, please specify the framework to "
                 "be loaded.");
  }
  }
  llvm::report_fatal_error("unhandled file format");
}

int main(int argc, char **argv) {
  // Register all our command line options.  We skipInitLLVM here because the
  // subframework will instantiate their own CLOptions instance and will this
  // call InitLLVM again.  Unfortunately InitLLVM does not allow multiple calls
  // to it, so we can't tolerate that.  Just avoid calling it here at the outer
  // level.
  GenericDriverCLOptions options(argc, argv, /*skipInitLLVM=*/true);

  llvm::cl::ParseCommandLineOptions(argc, argv, "Modular Tool (mt)\n");

  // If we don't have an input file (and we're attempting to run something that
  // doesn't take any input), then we have to pass in an empty memory buffer
  // which will result in an unknown file format. We want to preserve the stdin
  // behavior for cases where the command does take input because we might be
  // piping something into the tool.
  std::unique_ptr<MemoryBuffer> file = MemoryBuffer::getMemBuffer("", "empty");
  if (options.commandTakesInput() || options.inputFilename != "-")
    file = options.openInputFileOrExit();

  FileFormat format = options.getInputFormat(file->getBuffer());

  auto dylibOr = loadDesiredLibrary(argc, argv, *file, format, options.loadLib);
  if (failed(dylibOr))
    return options.reportError(dylibOr.getError());

  DriverAPI dylib = *dylibOr;

  // If the driver command is just `-get-version` then print the version and
  // exit right away. No need to do anything else.
  if (options.command == DriverCommand::getVersion) {
    llvm::outs() << "mt: " << MODULAR_VCS_VERSION
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

  // If we are not inspecting, we must need a compiled format.  If this is
  // already done then we're golden, otherwise compile now.
  if (format != FileFormat::kBEFFile)
    if (failed(dylib.compileToBinary(frameworkCtx, file, format)))
      return 1;

  // If we're stopping at compilation, emit the current BEF file to the
  // output.
  if (options.command == DriverCommand::compile) {
    auto outputFile = options.getOutputFile(options.commandHasBinaryOutput());
    outputFile->os().write(file->getBufferStart(), file->getBufferSize());
    outputFile->keep();
    return 0;
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
