//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMOGG_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMOGG_H

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

// Need to include the dialects we use, which includes many KGEN ones. These
// KGEN dialects are needed so we can load the LIT kernels to dynamically create
// the list of avaliable MOGG operations.
#include "Cache/CacheDialect/CacheDialect.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"

namespace M::MO {

/// Generate the code for declaring and registering the pass.
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "GenericML/GraphCompiler/Conversion/MOToMOGG.h.inc"

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMOGG_H
