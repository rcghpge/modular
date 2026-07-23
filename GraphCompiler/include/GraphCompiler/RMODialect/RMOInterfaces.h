//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the interface classes for the RMO dialect.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMOINTERFACES_H
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMOINTERFACES_H

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"

using M::KGEN::ParamDefValue;
using M::MO::TensorValue;

namespace M::RMO {
// Forward declare things used in RMOInterfaces.h.inc
class RMOOp;
LogicalResult verifyRMOOp(RMOOp op);
LogicalResult verifySameDTypeOperandsAndResults(RMOOp op);
} // namespace M::RMO

#include "GraphCompiler/RMODialect/RMOInterfaces.h.inc"

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOINTERFACES_H
