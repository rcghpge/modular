//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTRANSFORMS_H
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTRANSFORMS_H

#include "GraphCompiler/GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "mlir/Pass/Pass.h"

namespace M::MAP {

/// Generate the code for declaring the passes.
#define GEN_PASS_DECL
#include "GraphCompiler/GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "GraphCompiler/GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTRANSFORMS_H
