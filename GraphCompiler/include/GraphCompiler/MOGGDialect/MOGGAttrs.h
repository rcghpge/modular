//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the attribute classes for the MOGG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGATTRS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGATTRS_H

#include "GraphCompiler/MOGGDialect/MOGGEnums.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallSet.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h.inc"

namespace M::MOGG {

void printFusionInfo(mlir::AsmPrinter &p, mlir::Operation *op,
                     FusionInfoAttr fusionInfo);

mlir::ParseResult parseFusionInfo(mlir::AsmParser &p,
                                  FusionInfoAttr &fusionInfo);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGATTRS_H
