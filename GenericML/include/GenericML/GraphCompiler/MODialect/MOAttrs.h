//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace M::MO {
class ChainType;
class ShapeType;

/// Parse the parametric dimensions of a MO::ShapeAttr.
ParseResult parseShapeValue(AsmParser &p, SmallVector<TypedAttr> &values);

/// Parse the parametric dimensions of a MO::ShapeAttr.
ParseResult parseShapeValue(AsmParser &p, SmallVector<TypedAttr> &values,
                            MO::ShapeType type);

/// Print the parametric dimensions of a MO::ShapeAttr.
void printShapeValue(AsmPrinter &p, ArrayRef<TypedAttr> values,
                     MO::ShapeType type);

/// Print the parametric dimensions of a MO::ShapeAttr.
void printShapeValue(AsmPrinter &p, ArrayRef<TypedAttr> values);

} // namespace M::MO

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GenericML/GraphCompiler/MODialect/MOAttrs.h.inc"

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_H
