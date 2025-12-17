//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_GRTDIALECT_GRTCONTROLFLOWINTERFACE_H
#define GRAPHRT_GRTDIALECT_GRTCONTROLFLOWINTERFACE_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"

namespace M::GRT {

/// Indicates whether a control flow op is 'loop' like or 'condition' like.
enum ControlFlowFlavor { kLoopLike, kBranchLike };

// Verifies correctness of op implementing GRTControlFlowInterface.
LogicalResult verifyControlFlowInterfaceOp(Operation *op);

} // namespace M::GRT

#include "MLRT/GraphRT/GRTDialect/GRTControlFlowInterface.h.inc"

#endif // GRAPHRT_GRTDIALECT_GRTCONTROLFLOWINTERFACE_H
