//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPTRANSFORMS_H
#define GENERICML_MGPDIALECT_MGPTRANSFORMS_H

#include "Cache/CacheDialect/CacheDialect.h"
#include "Cache/CachePasses/CachePasses.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace M::MGP {

/// Generate the code for declaring the passes.
#define GEN_PASS_DECL
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

} // namespace M::MGP

#endif // GENERICML_MGPDIALECT_MGPTRANSFORMS_H
