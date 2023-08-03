//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/DriverInputFormat.h"
#include "Support/ML/CompiledFrameworkLabel.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <regex>

// We only use hasMEFMagicNumber, which is inlined in the header, so we don't
// have to link the MEF library.
#include "GraphRT/MEF/MEFEncoding.h"
#include "mlir/Bytecode/BytecodeReader.h"

using namespace M;

//===----------------------------------------------------------------------===//
// sniffInputFormat
//===----------------------------------------------------------------------===//

FileFormat M::sniffInputFormat(StringRef fileExtension,
                               StringRef fileContents) {

  // If the extension identifies the format, use that.
  if (fileExtension.contains("onnx")) {
    if (fileContents.contains("ir_version: "))
      return FileFormat::kONNXTextProtobuf;
    else
      return FileFormat::kONNXProtobuf;
  } else if (fileExtension.contains("pt")) {
    return FileFormat::kPTPickle;
  }

  // If there's nothing in the file, then we can't sniff anything.
  if (fileContents.empty())
    return FileFormat::kUnknown;

  // MEF files have a strong magic number we can depend on that self-identifies
  // them.
  if (MEF::hasMEFMagicNumber(fileContents))
    return FileFormat::kMEFFile;

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
    if (fileContents.contains("monnx.graph"))
      return FileFormat::kMONNXDialect;
    if (fileContents.contains("func.func") && fileContents.contains("tf."))
      return FileFormat::kTFDialect;
    if (fileContents.contains("operator_codes:"))
      return FileFormat::kTFLFlatBufferJSON;
    if (fileContents.contains("ir_version: "))
      return FileFormat::kONNXTextProtobuf;
    if (fileContents.contains("torch.debug_module_name"))
      return FileFormat::kTorchDialect;
    if (fileContents.contains("mo.graph"))
      return FileFormat::kMODialect;
    if (size_t start = fileContents.find("mgp.model");
        start != StringRef::npos) {
      // For mgp.model our desperation is even worse: we'll just look for
      //    mgp.model @<name> : <framework>
      StringRef suffix = fileContents.drop_front(start);
      std::regex regex(R"(mgp\.model\s+@\w+\s*:\s*(\w+))");
      std::match_results<StringRef::iterator> m;
      std::regex_search(suffix.begin(), suffix.end(), m, regex);
      if (!m.empty() && m.size() == 2) {
        // Map the CompiledFrameworkLabel enum into the larger FileFormat enum.
        switch (CompiledFrameworkLabel::getLabelForOpName("mgp.model",
                                                          StringRef(m[1]))
                    .value) {
        case CompiledFrameworkLabel::kUnknown:
          return FileFormat::kUnknown;
        case CompiledFrameworkLabel::kTensorFlowModel:
          return FileFormat::kTFPDialect;
        case CompiledFrameworkLabel::kTFLiteModel:
          return FileFormat::kTFLPDialect;
        case CompiledFrameworkLabel::kFauxModel:
          return FileFormat::kFauxDialect;
        case CompiledFrameworkLabel::kModularModel:
          return FileFormat::kMGPDialect;
        case CompiledFrameworkLabel::kONNXModel:
          return FileFormat::kMONNXDialect;
        case CompiledFrameworkLabel::kPyTorchModel:
          return FileFormat::kTorchPDialect;
        }
      }
    }
    if (fileContents.contains("faux.testcase"))
      return FileFormat::kFauxDialect;
  } else if (mlir::isBytecode(llvm::MemoryBufferRef(fileContents, ""))) {
    std::match_results<StringRef::iterator> m;
    if (std::regex_search(fileContents.begin(), fileContents.end(), m,
                          std::regex(R"(\bmtfl\.)"))) {
      return FileFormat::kMTFLDialect;
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
