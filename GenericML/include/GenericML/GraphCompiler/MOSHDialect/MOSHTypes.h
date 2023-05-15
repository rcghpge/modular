//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares types for the MOSH dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHTYPES_H
#define GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHTYPES_H

#include "KGEN/KGENDialect/KGENTypeInterfaces.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M::MOSH {
class ShapeType;
using ShapeValue = TypedValue<ShapeType>;
} // namespace M::MOSH

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h.inc"

#endif // GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHTYPES_H
