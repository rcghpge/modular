//===- BEFOps.h - Declare BEF dialect operations ----------------*- C++ -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the BEF dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_BEFDIALECT_BEFOPS_H
#define GRAPHRT_BEFDIALECT_BEFOPS_H

#include "GraphRT/BEFDialect/BEFDialect.h"
#include "mlir/IR/OpImplementation.h"

namespace M {
namespace BEF {
// TODO: Helpers here.
} // namespace BEF
} // namespace M

#define GET_OP_CLASSES
#include "GraphRT/BEFDialect/BEF.h.inc"

#endif // GRAPHRT_BEFDIALECT_BEFOPS_H
