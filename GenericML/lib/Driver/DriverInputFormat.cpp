//===- DriverInputFormat.cpp ----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/DriverInputFormat.h"
#include "GraphRT/BEFExecutor/BEFFile.h"
#include "LLCL/Support/RCRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace M;

//===----------------------------------------------------------------------===//
// sniffInputFormat
//===----------------------------------------------------------------------===//

FileFormat M::sniffInputFormat(StringRef fileContents) {
  // BEF files have a strong magic number we can depend on that self-identifies
  // them.
  if (isBEFFile(fileContents))
    return FileFormat::kBEFFile;

  // TFLite flatbuffer binary formats start with a 32-bit binary offset, then
  // have a four-byte "kFileIdentifier" field containing "TFL3".   This appears
  // to be true regardless of the version of the TFLite file, the version number
  // is stored elsewhere.
  if (fileContents.drop_front(4).take_front(4) == "TFL3")
    return FileFormat::kTFLFlatBuffer;

  // Otherwise, we're getting desperate: check to see if this looks like a
  // textual file of a familiar dialect.
  if (llvm::isASCII(fileContents)) {
    // TODO: We need something more sophisticated here, this will pick up
    // comments (e.g. CHECK lines), for example.
    if (fileContents.contains("tfl.graph"))
      return FileFormat::kTFLDialect;
    if (fileContents.contains("tflp.model"))
      return FileFormat::kTFLPDialect;
    if (fileContents.contains("faux.testcase"))
      return FileFormat::kFauxDialect;
    if (fileContents.contains("operator_codes:"))
      return FileFormat::kTFLFlatBufferJSON;
  }
  return FileFormat::kUnknown;
}

//===----------------------------------------------------------------------===//
// getFileContentIdentity
//===----------------------------------------------------------------------===//

ErrorOr<std::string>
M::getFileContentIdentity(const std::unique_ptr<llvm::MemoryBuffer> &file,
                          FileFormat format) {
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
    ErrorOr<RCRef<BEFFile>> befFileOr = BEFFile::open(file->getBuffer());
    if (failed(befFileOr))
      return Error(befFileOr.getError());

    auto befContents = befFileOr.get()->sniffFileContents();
    return std::string("input is a BEF file containing a ") +
           befContents.getAsString();
  }
  }
}
