//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_H

#include "GraphCompiler/GraphCompiler/MODialect/Transforms/ConstantFoldPass.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Pass/Pass.h"

// Must include dependent dialects
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace M::MO {

/// Generate the code for declaring the passes.
#define GEN_PASS_DECL
#include "GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

std::unique_ptr<Pass> createAddFallbackShapeFunctions(
    const M::MO::AddFallbackShapeFunctionsOptions &opts,
    std::shared_ptr<MOGG::MojoLibraryAnalysis> libraryAnalysis);

/// Creates the NanCheckPass. `stride > 1` samples one of every N float
/// kernel outputs; `stride == 1` checks every float kernel output (the
/// original pre-GEX-3703 behaviour, prohibitively slow on large models).
/// The default of 20 matches the `max-debug.nan-check-stride` knob.
std::unique_ptr<Pass>
createNanCheckPass(std::shared_ptr<MOGG::MojoLibraryAnalysis> libraryAnalysis,
                   uint64_t stride = 20);

} // namespace M::MO
//

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_H
