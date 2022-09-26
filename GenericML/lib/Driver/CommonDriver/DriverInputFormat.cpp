//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/DriverInputFormat.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

// We only use hasBEFMagicNumber, which is inlined in the header, so we don't
// have to link the BEF library.
#include "GraphRT/BEF/BEFEncoding.h"

using namespace M;

//===----------------------------------------------------------------------===//
// sniffInputFormat
//===----------------------------------------------------------------------===//

FileFormat M::sniffInputFormat(StringRef fileContents) {
  // If there's nothing in the file, then we can't sniff anything.
  if (fileContents.empty())
    return FileFormat::kUnknown;

  // BEF files have a strong magic number we can depend on that self-identifies
  // them.
  if (BEF::hasBEFMagicNumber(fileContents))
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
    if (fileContents.contains("mtfl.graph"))
      return FileFormat::kMTFLDialect;
    if (fileContents.contains("tflp.model"))
      return FileFormat::kTFLPDialect;
    if (fileContents.contains("faux.testcase"))
      return FileFormat::kFauxDialect;
    if (fileContents.contains("operator_codes:"))
      return FileFormat::kTFLFlatBufferJSON;
  }
  return FileFormat::kUnknown;
}

FileFormat
DriverInputFormatOptions::getInputFormat(StringRef fileContents) const {
  if (inputFormat != FileFormat::kUnknown)
    return inputFormat;
  return sniffInputFormat(fileContents);
}
