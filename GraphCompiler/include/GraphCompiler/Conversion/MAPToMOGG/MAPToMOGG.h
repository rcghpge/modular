//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MAPTOMOGG_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MAPTOMOGG_H

#include "mlir/Pass/Pass.h"

namespace M::MAP {

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "GraphCompiler/Conversion/MAPToMOGG/MAPToMOGG.h.inc"

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MAPTOMOGG_H
