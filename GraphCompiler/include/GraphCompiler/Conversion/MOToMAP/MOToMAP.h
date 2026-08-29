//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMAP_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMAP_H

#include "GraphCompiler/MODialect/MOUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace M::MO {

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "GraphCompiler/Conversion/MOToMAP/MOToMAP.h.inc"

// useAdvancedFusion() lives in MODialect/MOUtils.h -- MO-level code (e.g.
// DecomposeViewOps) needs it too, and must not depend on this file's own
// heavier deps (MAPDialect, MOGGDialect) just to check the env var.

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMAP_H
