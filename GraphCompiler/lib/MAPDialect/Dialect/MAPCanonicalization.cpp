//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace M::MAP;

namespace {

// Fusion can leave a `map.iter.opaque` binding the same value several times in
// `inputs` (once per fused producer). The `$inputs` list becomes the lowered
// kernel's parameter list, so keep it minimal: rewire every duplicate's
// shadowed block arg onto the first occurrence and erase it. Uniquing also
// makes the redundant loads and computations in the body identical, so CSE can
// collapse them.
struct DedupeIterOpaqueInputs : OpRewritePattern<IterOpaqueOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IterOpaqueOp op,
                                PatternRewriter &rewriter) const override {
    llvm::DenseMap<Value, BlockArgument> seen;
    llvm::SmallVector<unsigned> duplicates;
    for (auto [idx, input] : llvm::enumerate(op.getInputs()))
      if (!seen.insert({input, op.getInputBlockArgs()[idx]}).second)
        duplicates.push_back(idx);
    if (duplicates.empty())
      return failure();

    for (unsigned idx : duplicates)
      rewriter.replaceAllUsesWith(op.getInputBlockArgs()[idx],
                                  seen.lookup(op.getInputs()[idx]));
    rewriter.modifyOpInPlace(op, [&] {
      for (unsigned idx : llvm::reverse(duplicates))
        op.eraseInputsAt(idx);
    });
    return success();
  }
};

// Mirrors `DedupeIterOpaqueInputs`, one level further in: a capturing
// closure reaches its body through its own capture operand + shadowing
// block argument. Generic over the interface (one pattern, not one per
// closure type) since all three expose the same primitives. Also what
// `LoopInvariantViewMotion` relies on: it calls `addCapture` freely,
// expecting this to fold away any duplicate it creates.
struct DedupeCapturingClosureCaptures
    : OpInterfaceRewritePattern<CapturingFusionClosure> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(CapturingFusionClosure op,
                                PatternRewriter &rewriter) const override {
    Block::BlockArgListType captureArgs = op.getCaptureBlockArgs();

    llvm::DenseMap<Value, BlockArgument> seen;
    llvm::SmallVector<unsigned> duplicates;
    for (auto [idx, capture] : llvm::enumerate(op.getCaptures()))
      if (!seen.insert({capture, captureArgs[idx]}).second)
        duplicates.push_back(idx);
    if (duplicates.empty())
      return failure();

    for (unsigned idx : duplicates)
      rewriter.replaceAllUsesWith(captureArgs[idx],
                                  seen.lookup(op.getCaptures()[idx]));
    rewriter.modifyOpInPlace(op.getOperation(), [&] {
      for (unsigned idx : llvm::reverse(duplicates))
        op.eraseCaptureAt(idx);
    });
    return success();
  }
};

// Drops captures whose shadowed block arg has no use. Unlike
// `map.iter.opaque`'s own inputs below, a capture carries no graph-level
// scheduling edge of its own (that lives on the opaque's inputs), so
// dropping one outright is always safe -- what lets `LoopInvariantViewMotion`
// shed a capture a hoisted load no longer reads.
struct DropDeadCapturingClosureCaptures
    : OpInterfaceRewritePattern<CapturingFusionClosure> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(CapturingFusionClosure op,
                                PatternRewriter &rewriter) const override {
    Block::BlockArgListType captureArgs = op.getCaptureBlockArgs();
    llvm::SmallVector<unsigned> dead;
    for (auto [idx, arg] : llvm::enumerate(captureArgs))
      if (arg.use_empty())
        dead.push_back(idx);
    if (dead.empty())
      return failure();

    rewriter.modifyOpInPlace(op.getOperation(), [&] {
      for (unsigned idx : llvm::reverse(dead))
        op.eraseCaptureAt(idx);
    });
    return success();
  }
};

// Drops inputs whose shadowed block arg has no use, except chains: a chain
// input sequences the kernel against other memory ops at the graph level even
// when the body never consumes it, so dropping it would change scheduling.
struct DropDeadIterOpaqueInputs : OpRewritePattern<IterOpaqueOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IterOpaqueOp op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<unsigned> dead;
    for (auto [idx, arg] : llvm::enumerate(op.getInputBlockArgs()))
      if (arg.use_empty() && !llvm::isa<M::MO::ChainType>(arg.getType()))
        dead.push_back(idx);
    if (dead.empty())
      return failure();

    rewriter.modifyOpInPlace(op, [&] {
      for (unsigned idx : llvm::reverse(dead))
        op.eraseInputsAt(idx);
    });
    return success();
  }
};

} // namespace

void IterOpaqueOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<DedupeIterOpaqueInputs, DropDeadIterOpaqueInputs>(context);
}

void IterForeachBodyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.add<DedupeCapturingClosureCaptures, DropDeadCapturingClosureCaptures>(
      context);
}

void IterPrologueOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<DedupeCapturingClosureCaptures, DropDeadCapturingClosureCaptures>(
      context);
}

void IterEpilogueOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<DedupeCapturingClosureCaptures, DropDeadCapturingClosureCaptures>(
      context);
}
