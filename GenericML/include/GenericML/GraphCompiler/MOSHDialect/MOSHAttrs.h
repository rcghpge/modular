//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares attributes for the MOSH dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHATTRS_H
#define GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHATTRS_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/MDialect/MAttrs.h"

namespace M::MOSH {
class ChainType;
class ShapeType;
class ShapeAttr;

/// Return none if parser encountered something other than ShapeAttr or
/// UnknownAttr.
OptionalParseResult parseShapeAttrOrUnknown(AsmParser &p, TypedAttr &shapeAttr);

/// Print an attribute that has MOSH::ShapeType.
void printShapeTypedAttr(AsmPrinter &p, TypedAttr shapeAttr);

/// Helper to find the location of a parameter in a shape. Returns -1 if the
/// parameter does not appear in the shape.
int64_t findDeclIdx(ShapeAttr shape, KGEN::ParamDeclAttr decl);

/// Parse the parametric dimensions of a MOSH::ShapeAttr.
ParseResult parseShapeValue(AsmParser &p, SmallVector<TypedAttr> &values);

/// Parse the parametric dimensions of a MOSH::ShapeAttr.
ParseResult parseShapeValue(AsmParser &p, SmallVector<TypedAttr> &values,
                            MOSH::ShapeType type);

/// Print the parametric dimensions of a MOSH::ShapeAttr.
void printShapeValue(AsmPrinter &p, ArrayRef<TypedAttr> values,
                     MOSH::ShapeType type);

/// Print the parametric dimensions of a MOSH::ShapeAttr.
void printShapeValue(AsmPrinter &p, ArrayRef<TypedAttr> values);

} // namespace M::MOSH

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h.inc"

#endif // GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHATTRS_H
