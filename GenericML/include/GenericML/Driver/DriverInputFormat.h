//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_DRIVERINPUTFORMAT_H
#define GENERICML_DRIVER_DRIVERINPUTFORMAT_H

#include "Support/CommandLine.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

#include <filesystem>

namespace llvm {
class MemoryBuffer;
}

namespace M {

/// Format for a framework object.
/// This must be kept in sync with the Mojo `FrameworkFormat` enum.
enum class FrameworkFormat : uint8_t {
  MAXGraph = 0,
  TorchscriptModule = 1,
  TorchscriptFunction = 2,
  TorchMLIR = 3,
};

enum class FileFormat : uint8_t {
  kUnknown = 0,

  /// This looks like a MEF file of unknown contents.
  kMEFFile,

  /// This is a MLIR file with the mo dialect in it.
  kMODialect,

  /// This is a MLIR file with the mgp dialect and no fallbacks in it.
  kMGPDialect,

  /// This looks like a textual MLIR file with the Torch graph dialect in it.
  kTorchDialect,

  /// This looks like a PT file (TorchScript module) in pickled form.
  kPTPickle,

  /// This is a torchscript module object (not a file).
  kTorchscriptModule,

  /// This is a torchscript function object (not a file).
  kTorchscriptFunction,

  /// This looks like an MLIR file with PyTorch primitives in it.
  /// Those primitives may be from the ATenP or from the Torch dialect after
  /// lowering to MGP.
  kTorchPDialect,

  /// This is a MLIR file with the MO and MATen dialect in it.
  kTorchATenDialect,
};

inline FileFormat frameworkToFileFormat(FrameworkFormat frameworkFormat) {
  switch (frameworkFormat) {
  case FrameworkFormat::MAXGraph:
    return FileFormat::kMODialect;
  case FrameworkFormat::TorchscriptModule:
    return FileFormat::kTorchscriptModule;
  case FrameworkFormat::TorchscriptFunction:
    return FileFormat::kTorchscriptFunction;
  case FrameworkFormat::TorchMLIR:
    return FileFormat::kTorchDialect;
  }

  llvm_unreachable("unknown framework format");
}

/// Use heuristics to determine what kind of file the input file contents
/// represent.
FileFormat sniffInputFormat(StringRef fileExtension, StringRef fileContents);

struct DriverInputFormatOptions {
public:
  /// Allow the user to override the input file format.  The default is to
  /// autosense what the input is.
  FileFormat inputFormat{FileFormat::kUnknown};

  /// Return the input format the user specified, allowing them to specify
  /// something manually on the command line, but otherwise autosensing an
  /// answer.
  FileFormat getInputFormat(StringRef fileExtension,
                            StringRef fileContents) const;
};

class DriverInputFormatCLOptions {

public:
  DriverInputFormatOptions &options;
  DriverInputFormatCLOptions(DriverInputFormatOptions &opts) : options(opts) {}

private:
  llvm::cl::OptionCategory DriverInputFormatCategory{
      "Driver input format options"};
  /// Allow the user to override the input file format.  The default is to
  /// autosense what the input is.
  M::cl::MOpt<FileFormat, true> inputFormat{
      "input-format", cl::desc("input file format override"),
      cl::values(
          // Generic
          clEnumValN(FileFormat::kMODialect, "mo", "Input is a MO file"),
          clEnumValN(FileFormat::kMEFFile, "mef",
                     "Input is an executable MEF file"),
          clEnumValN(FileFormat::kMGPDialect, "mgp",
                     "Input is an mgp dialect file"),

          // PyTorch
          clEnumValN(FileFormat::kPTPickle, "torchscript",
                     "Input is a torchscript module file"),
          clEnumValN(FileFormat::kTorchscriptModule, "torchscript-module",
                     "Input is a torchscript module"),
          clEnumValN(FileFormat::kTorchscriptFunction, "torchscript-function",
                     "Input is a torchscript function"),
          clEnumValN(FileFormat::kTorchDialect, "torch",
                     "Input is a torch dialect file"),
          clEnumValN(FileFormat::kTorchATenDialect, "torch-aten",
                     "Input is a torch ATen dialect file"),
          clEnumValN(FileFormat::kTorchPDialect, "torch-p",
                     "Input is a torch primitives dialect file")),
      llvm::cl::location(options.inputFormat),
      llvm::cl::cat(DriverInputFormatCategory)};
};
} // namespace M

#endif // GENERICML_DRIVER_DRIVERINPUTFORMAT_H
