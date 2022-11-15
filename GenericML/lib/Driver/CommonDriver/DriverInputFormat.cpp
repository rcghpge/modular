//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/DriverInputFormat.h"
#include "GraphRT/BEF/BEFContents.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

#include <regex>

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
    // TODO(#4866): We need something more sophisticated here, this will pick up
    // comments (e.g. CHECK lines), for example.
    if (fileContents.contains("mtfl.graph"))
      return FileFormat::kMTFLDialect;
    if (size_t start = fileContents.find("mop.model");
        start != StringRef::npos) {
      // For mop.model our desperation is even worse: we'll just look for
      //    mop.model @<name> : <framework>
      StringRef suffix = fileContents.drop_front(start);
      std::regex regex(R"(mop\.model\s+@\w+\s*:\s*(\w+))");
      std::match_results<StringRef::iterator> m;
      std::regex_search(suffix.begin(), suffix.end(), m, regex);
      if (!m.empty() && m.size() == 2) {
        // Map the BEFContents enum into the larger FileFormat enum.
        switch (BEFContents::getContentsForOpName("mop.model", StringRef(m[1]))
                    .value) {
        case BEFContents::kUnknown:
          return FileFormat::kUnknown;
        case BEFContents::kTensorFlowModel:
          return FileFormat::kTFPDialect;
        case BEFContents::kTFLiteModel:
          return FileFormat::kTFLPDialect;
        case BEFContents::kFauxModel:
          return FileFormat::kFauxDialect;
        }
      }
    }
    if (fileContents.contains("tf."))
      // Top-level op in a tf graph is func.func, but matching against it
      // can cause ambiguity. So we check for the presence of any tf Ops.
      return FileFormat::kTFDialect;
    if (fileContents.contains("tfp.model"))
      return FileFormat::kTFPDialect;
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
