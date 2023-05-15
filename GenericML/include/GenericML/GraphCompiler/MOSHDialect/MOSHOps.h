//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the MOSH dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHOPS_H
#define GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHOPS_H

#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class RewriterBase;
} // namespace mlir

namespace M::MOSH {
class ShapeAttr;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/MOSHDialect/MOSH.h.inc"

#endif // GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHOPS_H
