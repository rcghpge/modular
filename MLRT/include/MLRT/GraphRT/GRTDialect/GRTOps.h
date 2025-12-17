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

#include "MLRT/GraphRT/GRTDialect/GRTChainInterface.h"
#include "MLRT/GraphRT/GRTDialect/GRTControlFlowInterface.h"
#include "MLRT/GraphRT/GRTDialect/GRTDialect.h"
#include "MLRT/GraphRT/MEFDialect/MEFNonStrictInterface.h"
#include "MLRT/GraphRT/MEFDialect/MEFOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "MLRT/GraphRT/GRTDialect/GRT.h.inc"

#endif // GRAPHRT_GRAPHRTDIALECT_GRTOPS_H
