//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares attributes for the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_H

// `MO::LayoutAttr` uses `mlir::StringAttr`
#include "GenericML/GraphCompiler/MODialect/MOEnums.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/ML/DType.h"
#include "mlir/IR/Attributes.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GenericML/GraphCompiler/MODialect/MOAttrs.h.inc"
#include "GenericML/GraphCompiler/MODialect/MOEnumAttrs.h.inc"

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_H
