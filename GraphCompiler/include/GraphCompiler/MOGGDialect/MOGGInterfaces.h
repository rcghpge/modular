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

namespace mlir {
class Operation;
} // namespace mlir

namespace M::MOGG {
// Shared verifier for `MOGG_TensorCreateInterface`, defined in MOGGOps.cpp
// where the MOGG types are in scope: enforces `!mogg.buffer_ => mut`. Called
// from the interface's generated `verifyTrait`, which cannot name the type.
::llvm::LogicalResult
verifyTensorCreateInterfaceMutability(::mlir::Operation *op);
} // namespace M::MOGG

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/MOGGInterfaces.h.inc"

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_H
