//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MODIALECT_MOTRANSFORMS_H
#define GENERICML_MODIALECT_MOTRANSFORMS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace M::MO {

/// Generate the code for declaring the passes.
#define GEN_PASS_DECL
#include "GenericML/MODialect/MOTransforms.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "GenericML/MODialect/MOTransforms.h.inc"

} // namespace M::MO

#endif // GENERICML_MODIALECT_MOTRANSFORMS_H
