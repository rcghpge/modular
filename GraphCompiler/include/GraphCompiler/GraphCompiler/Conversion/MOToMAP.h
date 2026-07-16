//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMAP_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMAP_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cstdlib>

namespace M::MO {

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "GraphCompiler/GraphCompiler/Conversion/MOToMAP.h.inc"

/// Returns whether the advanced (MAP/IMAP) fusion pipeline is enabled. Gated on
/// the presence of the `MAX_GC_USE_ADV_FUSION` environment variable; only its
/// existence matters, not its value.
inline bool useAdvancedFusion() {
  return std::getenv("MAX_GC_USE_ADV_FUSION") != nullptr;
}

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMAP_H
