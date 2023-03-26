//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_TRANSFORMS_MOGGPASSES_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_TRANSFORMS_MOGGPASSES_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

// Need to include the dialects we use, which includes many KGEN ones.
#include "Cache/CacheDialect/CacheDialect.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/InitAllDialects.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace M::MOGG {

// Return the lit registary, adding it if needed.
FailureOr<MOGG::LitRegisterOp> getOrLoadLitLibrary(ModuleOp mod);

//===----------------------------------------------------------------------===//
// Generated Pass Classes and Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_TRANSFORMS_MOGGPASSES_H
