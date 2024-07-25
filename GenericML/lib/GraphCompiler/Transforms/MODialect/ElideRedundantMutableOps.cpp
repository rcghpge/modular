//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MODevices.h"
#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <numeric>

#define DEBUG_TYPE "elide-redundant-store-load-pairs"

namespace M::MO {

#define GEN_PASS_DEF_ELIDEREDUNDANTMUTABLEOPS
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

namespace {

struct EliminateRedundantLoads
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MutableLoadOp> {

  EliminateRedundantLoads(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::MutableLoadOp loadOp,
                             DeviceAwareRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();

    auto inChain = loadOp.getInChain();

    if (!inChain.hasOneUse())
      return rewriter.notifyMatchFailure(loc,
                                         "input chain has more than one use");

    auto storeOp = dyn_cast_if_present<MutableStoreOp>(inChain.getDefiningOp());

    if (!storeOp)
      return rewriter.notifyMatchFailure(
          loc, "input chain not produced by a mo.mutable.store op");

    if (storeOp.getInBuffer() != loadOp.getInBuffer())
      return rewriter.notifyMatchFailure(
          loc, "preceding store op does not operate on the same mo.buffer");

    rewriter.replaceAllUsesWith(loadOp.getOutTensor(), storeOp.getInTensor());
    rewriter.replaceAllUsesWith(loadOp.getOutChain(), storeOp.getOutChain());

    rewriter.eraseOp(loadOp);

    return success();
  }
};

struct EliminateRedundantStores
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MutableStoreOp> {

  EliminateRedundantStores(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::MutableStoreOp currStoreOp,
                             DeviceAwareRewriter &rewriter) const override {
    auto loc = currStoreOp.getLoc();

    auto outChain = currStoreOp.getOutChain();

    if (!outChain.hasOneUse())
      return rewriter.notifyMatchFailure(loc,
                                         "output chain has more than one use");

    auto nextStoreOp = dyn_cast<MutableStoreOp>(*outChain.user_begin());

    if (!nextStoreOp)
      return rewriter.notifyMatchFailure(
          loc, "consumer of output chain is not another store op");

    if (currStoreOp.getInBuffer() != nextStoreOp.getInBuffer())
      return rewriter.notifyMatchFailure(
          loc, "following store op does not operate on same mo.buffer");

    rewriter.replaceAllUsesWith(currStoreOp.getOutChain(),
                                currStoreOp.getInChain());

    rewriter.eraseOp(currStoreOp);

    return success();
  }
};

} // namespace

struct ElideRedundantMutableOps
    : public impl::ElideRedundantMutableOpsBase<ElideRedundantMutableOps> {
  using ElideRedundantMutableOpsBase::ElideRedundantMutableOpsBase;

  void runOnOperation() override {
    auto graphOp = getOperation();
    MLIRContext *ctx = graphOp->getContext();
    OpBuilder builder{ctx};

    RewritePatternSet patterns{ctx};

    patterns.add<EliminateRedundantLoads, EliminateRedundantStores>(ctx);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace M::MO
