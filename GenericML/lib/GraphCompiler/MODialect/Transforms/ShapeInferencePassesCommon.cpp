//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/Transforms/ShapeInferencePassesCommon.h"
#include "mlir/Pass/PassManager.h"

using namespace M;
using namespace MO;

namespace M::MO::ShapeInferencePassCommon {

//===----------------------------------------------------------------------===//
// Common Patterns
//===----------------------------------------------------------------------===//

LogicalResult FoldConsecutiveRebinds::deviceAwareMatchAndRewrite(
    RebindOp op, DeviceAwareRewriter &rewriter) const {
  Location loc = op->getLoc();
  auto producerOp = op.getInput().getDefiningOp<RebindOp>();
  if (!producerOp) {
    return rewriter.notifyMatchFailure(loc, "not preceded by other mo.rebind");
  }

  rewriter.replaceOpWithNewOp<RebindOp>(op, op.getType(), producerOp.getInput(),
                                        op.getOutputParamDecls());
  return success();
}

} // namespace M::MO::ShapeInferencePassCommon
