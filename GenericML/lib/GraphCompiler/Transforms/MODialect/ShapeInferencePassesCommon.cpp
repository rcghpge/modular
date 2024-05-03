//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ShapeInferencePassesCommon.h"
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

  // Don't fold if producer input has unknown shape, because we want to
  // minimize rebind that takes input with `?` (shape propagation can't
  // propagate through `?`s). For instance,
  //
  // input : tensor<[?]>
  //
  //   r1 = rebind(input) : tensor<[?]>  -> tensor<[D1]>
  //   r2 = rebind(r1)    : tensor<[D1]> -> tensor<[D2]>
  // v.s.
  //   r1 = rebind(input) : tensor<[?]> -> tensor<[D1]>
  //   r2 = rebind(input) : tensor<[?]> -> tensor<[D2]>
  //
  // In the first case, shape propagation will understand that D1 == D2, but
  // in the second case, it will not.
  if (producerOp.getInput().getType().hasUnknownShape()) {
    return rewriter.notifyMatchFailure(
        loc, "preceding rebind has input with unknown shape");
  }

  rewriter.replaceOpWithNewOp<RebindOp>(op, op.getType(), producerOp.getInput(),
                                        op.getOutputParamDecls());
  return success();
}

} // namespace M::MO::ShapeInferencePassCommon
