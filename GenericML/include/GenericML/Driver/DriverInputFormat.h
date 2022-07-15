//===- GenericML/Driver/DriverInputFormat.h -------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_DRIVERINPUTFORMAT_H
#define GENERICML_DRIVER_DRIVERINPUTFORMAT_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

namespace llvm {
class MemoryBuffer;
}

namespace M {
enum class FileFormat {
  kUnknown = 0,

  /// This is a TFLite binary flatbuffer file.
  kTFLFlatBuffer,

  /// This is a TFLite flatbuffer file in JSON form.
  kTFLFlatBufferJSON,

  /// This looks like a textual MLIR file with the tfl dialect in it.
  kTFLDialect,

  /// This looks like a textual MLIR file with the tflp dialect in it.
  kTFLPDialect,

  /// This looks like a textual MLIR file with the faux dialect in it.
  kFauxDialect,

  /// This looks like a BEF file of unknown contents.
  kBEFFile
};

/// Use heuristics to determine what kind of file the input file contents
/// represent.
FileFormat sniffInputFormat(StringRef fileContents);
} // namespace M

#endif // GENERICML_DRIVER_DRIVERINPUTFORMAT_H
