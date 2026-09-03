//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/Support/TypeTraits.h"
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

// The largest constant worth rematerializing inside a kernel body rather than
// passing at runtime. Kept in step with `MOGG::FuseConstants`, which applies
// the same budget to the pre-advanced-fusion lane.
constexpr int64_t kMaxConstantSize = 2048;

// Returns the `mo.constant` producing `val` when its value can be
// rematerialized inside a kernel body, else null.
M::MO::ConstantOp getFusableConstant(Value val) {
  auto cst = val.getDefiningOp<M::MO::ConstantOp>();
  if (!cst || !M::MO::isOnHost(cst))
    return {};

  ElementsAttr elems = cst.getValueAttr();
  if (elems.getNumElements() > kMaxConstantSize)
    return {};
  if (!M::MOGG::TypeTraits::isSupportedMOGGType(elems.getElementType()))
    return {};
  if (!isa<M::ArrayElementsAttr, DenseResourceElementsAttr>(elems))
    return {};

  return cst;
}

// Rematerializes small constant inputs inside the body and drops them from
// `inputs`, which is what the lowered kernel's parameter list is built from.
// `map.iter.opaque` is isolated from above, so the clone has to live in the
// body; from there plain SSA dominance carries it to the loop body and to any
// prologue/epilogue capture.
//
// `DropDeadIterOpaqueInputs` would collect the operand on its own, and the
// original `mo.constant` is `Pure`, so the driver DCEs it once its last use
// goes. Clones left behind by folding the same constant into several input
// slots are uniqued by the folder, since `mo.constant` is `ConstantLike`.
struct FuseConstantIterOpaqueInputs : OpRewritePattern<IterOpaqueOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IterOpaqueOp op,
                                PatternRewriter &rewriter) const override {
    // Folding into a device kernel miscompiles it into illegal memory accesses
    // (GEX-3839).
    if (!M::MO::isOnHost(op.getOperation()))
      return failure();

    rewriter.setInsertionPointToStart(&op.getBody().front());
    llvm::SmallVector<unsigned> folded;
    for (auto [idx, input] : llvm::enumerate(op.getInputs())) {
      M::MO::ConstantOp cst = getFusableConstant(input);
      if (!cst)
        continue;
      // A `map.iter.opaque` body holds only MAP ops, so the value is carried
      // by a `map.constant` rather than a clone of the `mo.constant`.
      Value inBody = ConstantOp::create(rewriter, cst.getLoc(), cst.getType(),
                                        cst.getValueAttr());
      rewriter.replaceAllUsesWith(op.getInputBlockArgs()[idx], inBody);
      folded.push_back(idx);
    }
    if (folded.empty())
      return failure();

    // Erasing shifts the indices after it, so unwind back to front.
    rewriter.modifyOpInPlace(op, [&] {
      for (unsigned idx : llvm::reverse(folded))
        op.eraseInputsAt(idx);
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
  results.add<DedupeIterOpaqueInputs, DropDeadIterOpaqueInputs,
              FuseConstantIterOpaqueInputs>(context);
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
