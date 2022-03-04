//===- GraphRT/GRTDialect/GRTOps.h ------------------------------*- C++ -*-===//
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

#include "GraphRT/GRTDialect/GRTDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "GraphRT/GRTDialect/GRTTypes.h.inc"

#define GET_OP_CLASSES
#include "GraphRT/GRTDialect/GRT.h.inc"

#endif // GRAPHRT_GRAPHRTDIALECT_GRTOPS_H
