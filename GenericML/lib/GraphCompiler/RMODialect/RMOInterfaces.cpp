//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MO dialect interfaces.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/RMODialect/RMOInterfaces.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"

using namespace M;
using namespace RMO;

namespace M::RMO {
LogicalResult verifyRMOOp(RMOOp op) {
  auto parent = op->getParentOfType<MO::GraphOp>();
  if (!parent)
    return mlir::emitError(op.getLoc(), "Do not have a parent graph op!");

  return success();
}
} // namespace M::RMO

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/RMODialect/RMOInterfaces.cpp.inc"
