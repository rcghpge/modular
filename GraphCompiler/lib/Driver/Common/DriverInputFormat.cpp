//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Driver/Common/DriverInputFormat.h"
#include "Support/ML/CompiledFrameworkLabel.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <regex>

// We only use hasMEFMagicNumber, which is inlined in the header, so we don't
// have to link the MEF library.
#include "MLRT/GraphRT/MEF/MEFEncoding.h"
#include "mlir/Bytecode/BytecodeReader.h"

using namespace M;

//===----------------------------------------------------------------------===//
// sniffInputFormat
//===----------------------------------------------------------------------===//

FileFormat M::sniffInputFormat(StringRef fileExtension,
                               StringRef fileContents) {

  // If there's nothing in the file, then we can't sniff anything.
  if (fileContents.empty())
    return FileFormat::kUnknown;

  // MEF files have a strong magic number we can depend on that self-identifies
  // them.
  if (MEF::hasMEFMagicNumber(fileContents))
    return FileFormat::kMEFFile;

  // Otherwise, we're getting desperate: check to see if this looks like a
  // textual file of a familiar dialect.
  if (llvm::isASCII(fileContents)) {
    if (fileContents.contains("mo.graph"))
      return FileFormat::kMODialect;
    if (size_t start = fileContents.find("mef.model");
        start != StringRef::npos) {
      // For mef.model our desperation is even worse: we'll just look for
      //    mef.model @<name> : <framework>
      StringRef suffix = fileContents.drop_front(start);
      std::regex regex(R"(mef\.model\s+@\w+\s*:\s*(\w+))");
      std::match_results<StringRef::iterator> m;
      std::regex_search(suffix.begin(), suffix.end(), m, regex);
      if (!m.empty() && m.size() == 2) {
        // Map the CompiledFrameworkLabel enum into the larger FileFormat enum.
        switch (CompiledFrameworkLabel::getLabelForOpName("mef.model",
                                                          StringRef(m[1]))
                    .value) {
        case CompiledFrameworkLabel::kUnknown:
          return FileFormat::kUnknown;
        case CompiledFrameworkLabel::kModularModel:
          return FileFormat::kMGPDialect;
        }
      }
    }
  } else if (mlir::isBytecode(llvm::MemoryBufferRef(fileContents, ""))) {
    // Sniff the MLIR bytecode looking for the dialect id on a word boundary.
    // TODO(#26701): Replace this with a reliable / fast dialect ID encoding.
    constexpr int maxSniffedBytes = 1024 * 1024;
    std::match_results<StringRef::iterator> m;
    if (std::regex_search(fileContents.begin(),
                          std::min(fileContents.end(),
                                   fileContents.begin() + maxSniffedBytes),
                          m, std::regex(R"(\bmo\.)"))) {
      return FileFormat::kMODialect;
    }
  }
  return FileFormat::kUnknown;
}

FileFormat
DriverInputFormatOptions::getInputFormat(StringRef fileExtension,
                                         StringRef fileContents) const {
  if (inputFormat != FileFormat::kUnknown)
    return inputFormat;
  return sniffInputFormat(fileExtension, fileContents);
}
