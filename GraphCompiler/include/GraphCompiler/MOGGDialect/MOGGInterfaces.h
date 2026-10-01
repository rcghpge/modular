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

// `MOGGTypes.h` (for `!mogg.buffer`/`!mogg.tensor`) and `Builders.h` (for
// `OpBuilder::clone`) are needed by the `cloneWithMut` default implementation,
// which is emitted inline into the generated interface header below. Neither
// pulls in `MOGGInterfaces.h`, so there is no include cycle.
#include "GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "mlir/IR/Builders.h"

namespace M::MOGG {
// Shared verifier for `MOGG_TensorCreateInterface`, defined in MOGGOps.cpp
// where the MOGG types are in scope: enforces `!mogg.buffer => mut`. Called
// from the interface's generated `verifyTrait`, which cannot name the type.
::llvm::LogicalResult
verifyTensorCreateInterfaceMutability(::mlir::Operation *op);
} // namespace M::MOGG

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/MOGGInterfaces.h.inc"

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_H
