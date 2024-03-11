//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the RMO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_H
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_H

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/RMODialect/RMOInterfaces.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "mlir/IR/OpImplementation.h"

using M::KGEN::ParamDeclAttr;
using M::MOSH::ShapeAttr;

namespace M::RMO {

/// Rebinds all values
SmallVector<Value> rebindUnknownShapeInValues(ValueRange inputValues,
                                              MO::GraphOp graphOp,
                                              OpBuilder &builder);
} // namespace M::RMO
//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/RMODialect/RMO.h.inc"

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_H
