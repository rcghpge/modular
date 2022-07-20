//===- mt.cpp -------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/DriverAPI.h"
#include "GenericML/Driver/DriverCLOptions.h"
#include "GenericML/Support/ModelCollection.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ToolOutputFile.h"

#include "PlatformDylib.h.inc"
#include "llvm/Support/InitLLVM.h"

using namespace M;
using llvm::MemoryBuffer;

static constexpr char tflLibraryName[] = SYSTEM_SHARED_LIBRARY("tfl");

/// Get a string that describes the contents of the BEF file. Given that this
/// driver cannot link the BEF libraries, we have to rely on the symbols
/// provided by the framework.
static ErrorOr<std::string>
getFileContentIdentity(const llvm::MemoryBuffer &file, FileFormat format,
                       decltype(M_openBinaryFile) openHandle,
                       decltype(M_sniffBinaryFileContents) sniffHandle) {
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
    ModelBinaryHandle handle;
    if (auto err = openHandle(file, &handle))
      return err.takeError();

    std::string out;
    sniffHandle(handle, &out);
    return out;
  }
  }
}

static ErrorOr<llvm::sys::DynamicLibrary>
loadDesiredLibrary(const llvm::MemoryBuffer &file, FileFormat inputFormat,
                   StringRef forceLoad) {
  std::string errorMsg;

  // Wrap getPermanentLibrary with a thing that extracts the error message and
  // returns it as an Error.
  auto fallibleLibraryLoad =
      [&](StringRef name) -> ErrorOr<llvm::sys::DynamicLibrary> {
    auto dylib =
        llvm::sys::DynamicLibrary::getPermanentLibrary(name.data(), &errorMsg);
    if (!dylib.isValid())
      return Error("failed to load library " + name + ": " + errorMsg);
    return dylib;
  };

  // If we have been forced to load a specific lib, just load it and be done.
  if (!forceLoad.empty())
    return fallibleLibraryLoad(forceLoad);

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
    auto openBEFFileHandle =
        (decltype(&M_openBinaryFile))tflDylib.getAddressOfSymbol(
            "M_openBinaryFile");
    auto sniffBEFFileContentsHandle =
        (decltype(&M_sniffBinaryFileContents))tflDylib.getAddressOfSymbol(
            "M_sniffBinaryFileContents");

    ModelBinaryHandle handle;
    if (auto err = openBEFFileHandle(file, &handle))
      return err.takeError();

    std::string contents;
    sniffBEFFileContentsHandle(handle, &contents);
    if (StringRef(contents).contains_insensitive("tflite"))
      return tflDylib;

    if (StringRef(contents).contains_insensitive("faux"))
      return Error("Faux is not yet supported.");

    return Error("unknown BEF file contents, please specify the framework to "
                 "be loaded.");
  }
  }
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Register all our command line options.
  DriverCLOptions options(argv[0]);

  llvm::cl::ParseCommandLineOptions(argc, argv, "Modular Tool (mt)\n");

  // Open the input file, and determine what it is.
  std::unique_ptr<MemoryBuffer> file = options.openInputFileOrExit();
  FileFormat format = options.getInputFormat(file->getBuffer());

  auto dylibOr = loadDesiredLibrary(*file, format, options.loadLib);
  if (failed(dylibOr))
    return options.reportError(dylibOr.getError());

  llvm::sys::DynamicLibrary dylib = *dylibOr;

  // Pull out the handles we want.
  auto compileToBEFHandle =
      (decltype(&M_compileToBinary))dylib.getAddressOfSymbol(
          "M_compileToBinary");
  auto getMetadataFromBEFHandle =
      (decltype(&M_getMetadataFromBinary))dylib.getAddressOfSymbol(
          "M_getMetadataFromBinary");
  auto checkExpectedContentsHandle =
      (decltype(&M_checkExpectedContents))dylib.getAddressOfSymbol(
          "M_checkExpectedContents");
  auto openBinaryFileHandle =
      (decltype(&M_openBinaryFile))dylib.getAddressOfSymbol("M_openBinaryFile");
  auto sniffBinaryFileContentsHandle =
      (decltype(&M_sniffBinaryFileContents))dylib.getAddressOfSymbol(
          "M_sniffBinaryFileContents");
  auto runBinaryFileHandle =
      (decltype(&M_runBinaryFile))dylib.getAddressOfSymbol("M_runBinaryFile");

  // Start processing the input in stages.  The inspect command is first,
  // because it works on any input without transformation.
  if (options.command == DriverCommand::inspect) {
    std::unique_ptr<llvm::ToolOutputFile> outputFile = options.getOutputFile();
    auto identityOr = getFileContentIdentity(
        *file, format, openBinaryFileHandle, sniffBinaryFileContentsHandle);
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
    if (failed(compileToBEFHandle(file, format, options)))
      return 1;

  // If we're stopping at compilation, emit the current BEF file to the
  // output.
  if (options.command == DriverCommand::compile) {
    auto outputFile = options.getOutputFile();
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

  ModelBinaryHandle handle;
  if (auto err = openBinaryFileHandle(*file, &handle))
    return options.reportError(err.getError());

  // Check to see if this file contains what we expected.  If not, throw an
  // error message.
  if (auto err = checkExpectedContentsHandle(handle))
    return options.reportError(err.getError());

  // If we're just getting information, then get the BEF information
  // and print it.
  if (options.command == DriverCommand::printMetadata ||
      options.command == DriverCommand::printConstantMetadata) {
    if (auto err = getMetadataFromBEFHandle(handle, options))
      return options.reportError(err.getError());
    return 0;
  }

  // Hand off the BEFFileHandle and the file buffer to the framework to actually
  // run the thing with all the various diagnostics.
  return runBinaryFileHandle(std::move(file), handle, options);
}
