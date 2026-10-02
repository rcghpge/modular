//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGINTERFACES_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGINTERFACES_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/Support/LogicalResult.h"

// Needed by the inline `cloneWithMut` default implementation in the generated
// header. Neither includes `MOGGInterfaces.h`, so there is no cycle.
#include "GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "mlir/IR/Builders.h"

namespace M::MOGG {
// Enforces `!mogg.buffer => mut` for `MOGG_TensorCreateInterface`. Defined in
// MOGGOps.cpp because the generated `verifyTrait` cannot name the MOGG types.
::llvm::LogicalResult
verifyTensorCreateInterfaceMutability(::mlir::Operation *op);
} // namespace M::MOGG

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/MOGGInterfaces.h.inc"

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_H
