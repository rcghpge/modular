//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEVIEWS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

/**
1. Fuse compatible view ops into other kernels.
2. Lower `mo.static.reshape` ops into a kernel and fuse them into other kernels.
The reason this is done as a separate step is because reshape ops are not
lowered and they must be the last view op fused into a kernel because the kernel
implementation expects contiguous strides. Therefore, bcast -> reshape will
yield invalid results.
3. Raise mogg.kernels that used to be mo.static.reshape back into reshapes. This
step is optional for correctness but necessary for performance. Reshapes, as
with other views, copy data. A later pass, MOToPrimitives, will turn the reshape
into a primitive that does not copy.

All in all, this pass fuses views into other views or kernels that aren't
reshapes. Then, reshapes are fused and the remaining ones will be later lowered
to a fast primitive.

GUARANTEES:
- All remaining reshapes are `mo.static.reshape.
- No view fuses into a reshape kernel.
 */

bool isUnFusedReshapeKernel(Operation *op) {
  auto kernel = dyn_cast<MOGG::ExpKernelOp>(op);
  if (!kernel || !kernel.isViewOp())
    return false;

  SmallVector<CallViewOp> callViews =
      llvm::to_vector(kernel.getExecBody().getOps<CallViewOp>());

  // This is a unfused reshape operations.
  return callViews.size() == 1 && callViews.front().getKernelName() ==
                                      MO::Static::ReshapeOp::getOperationName();
}

void raiseBackReshapeKernels(MLIRContext *ctx,
                             ArrayRef<ExpKernelOp> reshapeKernels) {
  OpBuilder builder(ctx);
  for (auto k : reshapeKernels) {
    builder.setInsertionPointAfter(k);
    auto reshapeOp = MO::Static::ReshapeOp::create(
        builder, k->getLoc(), k->getResultTypes(), k.getOperands());

    k->replaceAllUsesWith(reshapeOp);
    k->erase();
  }
}

struct FuseViewsPass : public impl::FuseViewsBase<FuseViewsPass> {
  using FuseViewsBase::FuseViewsBase;

  FuseViewsPass() = default;
  void runOnOperationImpl(MO::GraphOp graph, MLIRContext *ctx,
                          bool fuseReshape) {
    graph.walk([&](MOGG::ExpKernelOp op) {
      // We never fuse other kernel into reshapes since reshape requires a
      // contiguous buffer.
      if (isUnFusedReshapeKernel(op))
        return;

      // We are fusing views, which won't be blocked if there is already other
      // view fused into the kernel.
      if (!op.hasFusibleOperand(/*lookThroughView=*/true) && !op.isViewOp())
        return;

      // Traverse from the back to simplify removing operands.
      int32_t index = op->getNumOperands() - 1;
      while (index >= 0) {
        if (!canFuseOperand(index, op)) {
          index--;
          continue;
        }

        bool isReshape =
            isUnFusedReshapeKernel((op->getOperand(index).getDefiningOp()));
        // if we are not fusing in reshapes, skip fusible reshape operand.
        // If we are fusing in reshapes, then we skip things that aren't
        // reshapes. This avoids cases like:
        //   non-reshape -> reshape -> non-reshape
        //   non-reshape -> reshape+non-reshape (first iteration)
        //   non-reshape+reshape+non-reshape  (second iteration)
        if (fuseReshape != isReshape) {
          index--;
          continue;
        }

        fuseInputAtIndex(op, index);
        index = op->getNumOperands() - 1;
      }
    });
  }

  /// Return the index of the input to fuse or -1 if nothing can be fused.
  bool canFuseOperand(int32_t index, Operation *op) {
    // Block arguments have no defining op and obviously cant be fused.
    Operation *edge = op->getOperand(index).getDefiningOp();
    if (!edge)
      return false;

    // Only fuse within the same block and not across, i.e, control flow.
    if (edge->getBlock() != op->getBlock())
      return false;

    if (auto expKernel = dyn_cast<MOGG::ExpKernelOp>(op)) {
      if (auto alloc = expKernel.getAllocIndices(); alloc && !alloc.empty())
        return false;

      auto otherKernel = dyn_cast<MOGG::ExpKernelOp>(edge);
      if (!otherKernel)
        return false;

      // Only fuse view operations (when will a view kernel returns more than
      // one result, probably dead code, or should we instead verify it during
      // IR construction?)
      if (!otherKernel.isViewOp() || otherKernel.getNumResults() != 1)
        return false;

      // if the kernel is not a view kernel nor has a view fusible input at the
      // given position.
      if (!expKernel.isViewOp() && !expKernel.isOperandFusible(index, true))
        return false;

      return true;
    }

    return false;
  }

  void fuseInputAtIndex(MOGG::ExpKernelOp kernel, int32_t idx) {
    ++numberOfFusedViews;

    Value toFuse = kernel.getOperand(idx);
    auto viewProducer = cast<MOGG::ExpKernelOp>(toFuse.getDefiningOp());
    kernel.insertInputsAt(idx + 1, viewProducer.getInputs());

    IRRewriter rewriter(kernel->getContext());
    rewriter.cloneRegionBefore(viewProducer.getExecBody(),
                               &kernel.getExecBody().front());

    Block *src = &kernel.getExecBody().front();
    Block *dst = &kernel.getExecBody().back();
    auto srcTerm = cast<MOGG::CallMaterializeOp>(src->getTerminator());

    ValueRange inlinedArgs = dst->getArguments().slice(
        idx + 1, viewProducer.getExecBody().getNumArguments());

    rewriter.inlineBlockBefore(src, dst, dst->begin(), inlinedArgs);

    // Clean up.
    dst->getArgument(idx).replaceAllUsesWith(srcTerm.getInput());
    kernel.eraseInputsAt(idx);
    auto srcOutputPlaceholders = srcTerm.getOutput().getDefiningOp();
    srcTerm.erase();
    srcOutputPlaceholders->erase();

    kernel.setSummaryAttr(
        getSummaryString({viewProducer, kernel}, FusionType::HARD));
  }

  void runOnOperation() override {
    auto graph = getOperation();
    auto ctx = graph.getContext();

    // First fuse all kernels that are view ops.
    runOnOperationImpl(graph, ctx, /*fuseReshape=*/false);

    // The fuse reshapes
    SmallVector<ExpKernelOp> reshapes;
    graph.walk([&](MOGG::ExpKernelOp op) {
      if (isUnFusedReshapeKernel(op))
        reshapes.push_back(op);
    });
    runOnOperationImpl(graph, ctx, /*fuseReshape=*/true);

    // Finally, raise unfused reshapes back to a MO op to be eventually lowered
    // as MGP primitives (a no-op, really).
    raiseBackReshapeKernels(ctx, reshapes);
  }
};

} // namespace

} // namespace M::MOGG
