//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Pass/Pass.h"

// Must include dependent dialects
#include "Cache/CacheDialect/CacheDialect.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace M::MO {

/// Generate the code for declaring the passes.
#define GEN_PASS_DECL
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_H
