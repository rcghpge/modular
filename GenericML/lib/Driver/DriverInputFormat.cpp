//===- DriverInputFormat.cpp ----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/DriverInputFormat.h"
#include "GraphRT/BEFExecutor/BEFFile.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

using namespace M;

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
