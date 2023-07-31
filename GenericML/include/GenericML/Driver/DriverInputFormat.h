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

namespace llvm {
class MemoryBuffer;
}

namespace M {
enum class FileFormat {
  kUnknown = 0,

  /// This looks like a MEF file of unknown contents.
  kMEFFile,

  /// This looks like a textual MLIR file with the faux dialect in it.
  kFauxDialect,

  /// This looks like a textual MLIR file with the tfl dialect in it.
  kMTFLDialect,

  /// This is a TFLite binary flatbuffer file.
  kTFLFlatBuffer,

  /// This is a TFLite flatbuffer file in JSON form.
  kTFLFlatBufferJSON,

  /// This looks like a textual MLIR file with the tflp dialect in it.
  kTFLPDialect,

  /// This looks like a textual MLIR file with the tfp dialect in it.
  kTFPDialect,

  /// This looks like a textual MLIR file with the tensorflow graph dialect in
  /// it.
  kTFDialect,

  /// This is a TensorFlow SavedModel directory.
  kTFSavedModel,

  /// This is a MLIR file with the mo dialect in it.
  kMODialect,

  /// This is a MLIR file with the mgp dialect and no fallbacks in it.
  kMGPDialect,

  /// This looks like a textual MLIR file with the MONNX graph dialect in it.
  kMONNXDialect,

  /// This looks like an ONNX binary protobuf.
  kONNXProtobuf,

  /// This looks like an ONNX textual protobuf.
  kONNXTextProtobuf,

  /// This looks like a textual MLIR file with the Torch graph dialect in it.
  kTorchDialect,

  /// This looks like a PT file (TorchScript module) in pickled form.
  kPTPickle,

  /// This looks like an MLIR file with PyTorch primitives in it.
  /// Those primitives may be from the ATenP or from the Torch dialect after
  /// lowering to MGP.
  kTorchPDialect,
};

/// Helper to check if a format is within the valid range.
inline bool isValidFileFormat(FileFormat format) {
  return (format >= FileFormat::kUnknown) &&
         (format <= FileFormat::kTorchPDialect);
}

/// Use heuristics to determine what kind of file the input file contents
/// represent.
FileFormat sniffInputFormat(StringRef fileExtension, StringRef fileContents);

class DriverInputFormatOptions {
public:
  /// Allow the user to override the input file format.  The default is to
  /// autosense what the input is.
  cl::opt<FileFormat> inputFormat{
      "input-format", cl::desc("input file format override"),
      cl::values(clEnumValN(FileFormat::kTFLFlatBuffer, "tflite-flat-buffer",
                            "Input is a TFLite flatbuffer file"),
                 clEnumValN(FileFormat::kMODialect, "mo", "Input is a MO file"),
                 clEnumValN(FileFormat::kMEFFile, "mef",
                            "Input is an executable MEF file"),
                 clEnumValN(FileFormat::kTFSavedModel, "tf-saved-model",
                            "Input is a TF SavedModel directory"),
                 clEnumValN(FileFormat::kONNXProtobuf, "onnx-proto",
                            "Input is an ONNX model protobuf file")),
      cl::init(FileFormat::kUnknown)};

  /// Return the input format the user specified, allowing them to specify
  /// something manually on the command line, but otherwise autosensing an
  /// answer.
  FileFormat getInputFormat(StringRef fileExtension,
                            StringRef fileContents) const;
};
} // namespace M

#endif // GENERICML_DRIVER_DRIVERINPUTFORMAT_H
