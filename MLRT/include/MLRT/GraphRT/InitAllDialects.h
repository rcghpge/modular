//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file registers all the dialects in the GraphRT library.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_INITALLDIALECTS_H
#define MLRT_GRAPHRT_INITALLDIALECTS_H

#include "MLRT/GraphRT/GRTDialect/GRTDialect.h"
#include "MLRT/GraphRT/MEFDialect/MEFDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M {
// Add all the MLIR dialects to the provided registry.
inline void registerAllGraphRTDialects(DialectRegistry &registry) {
  registry.insert<MEF::MEFDialect, GRT::GRTDialect>();
}

} // namespace M

#endif // MLRT_GRAPHRT_INITALLDIALECTS_H
