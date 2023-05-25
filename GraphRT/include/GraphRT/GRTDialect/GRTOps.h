//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the GRT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_GRAPHRTDIALECT_GRTOPS_H
#define GRAPHRT_GRAPHRTDIALECT_GRTOPS_H

#include "GraphRT/GRTDialect/GRTContolFlowInterface.h"
#include "GraphRT/GRTDialect/GRTDialect.h"
#include "GraphRT/MEFDialect/MEFNonStrict.h"
#include "GraphRT/MEFDialect/MEFOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "GraphRT/GRTDialect/GRT.h.inc"

#endif // GRAPHRT_GRAPHRTDIALECT_GRTOPS_H
