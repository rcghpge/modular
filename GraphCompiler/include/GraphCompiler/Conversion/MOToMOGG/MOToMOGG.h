//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMOGG_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMOGG_H

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
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

namespace M::MO {

/// Generate the code for declaring and registering the pass.
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "GraphCompiler/Conversion/MOToMOGG/MOToMOGG.h.inc"

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMOGG_H
