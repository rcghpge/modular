//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares transforms for the RMO dialect.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMOTRANSFORMS_H
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMOTRANSFORMS_H

#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/RMODialect/RMODialect.h"
#include "mlir/Pass/Pass.h"

using namespace M::MO;

namespace mlir {
class Pass;
} // namespace mlir

namespace M::RMO {

/// Generate the code for declaring the passes.
#define GEN_PASS_DECL
#include "GraphCompiler/RMODialect/Transforms/RMOTransforms.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "GraphCompiler/RMODialect/Transforms/RMOTransforms.h.inc"

} // namespace M::RMO

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOTRANSFORMS_H
