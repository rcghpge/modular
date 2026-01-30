//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOTOPRIMITIVES_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOTOPRIMITIVES_H

#include "Support/AlignedAlloc.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

// Need to include the dialects we use, which includes many KGEN ones. These
// KGEN dialects are needed so we can load the LIT kernels to dynamically create
// the list of available MOGG operations.
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

namespace M::MGP {

/// Generate the code for declaring the pass.
#define GEN_PASS_DECL_MOTOPRIMITIVES
#define GEN_PASS_DECL_LEGALIZEASYNCREFDEALLOCATIONS
#define GEN_PASS_DECL_RAISEBACKINDEXCONSTANT
#define GEN_PASS_DECL_LOWERPRIMITIVES
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

/// Generate the code for registering the pass.
#define GEN_PASS_REGISTRATION
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

namespace M::MOGG {
class MojoLibraryAnalysis;
} // namespace M::MOGG

namespace M::MGP {

std::unique_ptr<mlir::Pass> createLowerPrimitives(
    const LowerPrimitivesOptions &opts,
    std::shared_ptr<MOGG::MojoLibraryAnalysis> libraryAnalysis);

} // namespace M::MGP

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOTOPRIMITIVES_H
