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

// A prologue/epilogue region is isolated from above, so a value reaches its
// body through a second boundary of its own: the capture operand and the inner
// block argument shadowing it. Uniquing them mirrors the input uniquing --
// deduping `iter.opaque` inputs is what turns captures of previously-distinct
// block args into duplicates of each other, and the driver's fixed point
// sequences the two automatically.
template <typename CallbackOpT>
struct DedupeCallbackCaptures : OpRewritePattern<CallbackOpT> {
  using OpRewritePattern<CallbackOpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(CallbackOpT op,
                                PatternRewriter &rewriter) const override {
    Block &body = op.getBody().front();
    // The leading block args (the index, plus the stored value for an
    // epilogue) precede the captures.
    unsigned offset = body.getNumArguments() - op.getCaptures().size();

    llvm::DenseMap<Value, BlockArgument> seen;
    llvm::SmallVector<unsigned> duplicates;
    for (auto [idx, capture] : llvm::enumerate(op.getCaptures()))
      if (!seen.insert({capture, body.getArgument(offset + idx)}).second)
        duplicates.push_back(idx);
    if (duplicates.empty())
      return failure();

    for (unsigned idx : duplicates)
      rewriter.replaceAllUsesWith(body.getArgument(offset + idx),
                                  seen.lookup(op.getCaptures()[idx]));
    rewriter.modifyOpInPlace(op, [&] {
      for (unsigned idx : llvm::reverse(duplicates)) {
        op.getCapturesMutable().erase(idx);
        body.eraseArgument(offset + idx);
      }
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

void IterPrologueOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<DedupeCallbackCaptures<IterPrologueOp>>(context);
}

void IterEpilogueOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<DedupeCallbackCaptures<IterEpilogueOp>>(context);
}
