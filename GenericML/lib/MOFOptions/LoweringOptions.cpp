//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MOFOptions.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;
using namespace M::MOF;

void MOFLoweringOptions::setupContext(M::ContextRef context) {
  context->emplaceIfMissing<DebugTensorPrintOptions>(debugTensorPrintOptions);
}

void MOFLoweringOptions::print(llvm::raw_ostream &os) {
  os << "debugTensorPrintOptions={";
  os << "style=";

#define PRINT_ENUM_VALUES(VALUE)                                               \
  case M::ResultOutputStyle::VALUE:                                            \
    os << #VALUE;                                                              \
    break

  switch (debugTensorPrintOptions.style) {
    PRINT_ENUM_VALUES(kCompact);
    PRINT_ENUM_VALUES(kBinary);
    PRINT_ENUM_VALUES(kBinaryMaxCheckpoint);
    PRINT_ENUM_VALUES(kFull);
    PRINT_ENUM_VALUES(kNone);
  }

#undef PRINT_ENUM_VALUES

  os << ";precision=" << debugTensorPrintOptions.precision << ";";
  os << "binaryDir=" << debugTensorPrintOptions.binaryDir << ";";
  os << "};";
}
