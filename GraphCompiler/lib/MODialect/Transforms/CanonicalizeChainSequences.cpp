//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "Support/AssertStream.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetOperations.h"

namespace M::MO {

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_CANONICALIZECHAINSEQUENCES
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

using namespace M;
using namespace M::MO;

static bool hasSingleChain(MOMutableOpInterface mutOp) {
  return mutOp.getInChains().size() == 1 && mutOp.getOutChains().size() == 1;
}

/// Collects operands with write effects for a mutable op. If effects cannot be
/// determined (e.g. the op or a nested op doesn't implement
/// MemoryEffectOpInterface), conservatively assumes all mutable operands are
/// written. This ensures chain relaxation is never applied when write effects
/// are unknown.
static llvm::SmallDenseSet<Value>
collectWriteEffectOperands(MOMutableOpInterface mutOp) {
  std::optional<llvm::SmallVector<mlir::MemoryEffects::EffectInstance>>
      effects = mlir::getEffectsRecursively(mutOp);

  if (!effects) {
    // Cannot determine effects — conservatively treat all mutable operands as
    // written so that chain relaxation is blocked.
    auto range =
        llvm::make_filter_range(mutOp->getOperands(), isMutableOperand);
    return {range.begin(), range.end()};
  }

  llvm::SmallDenseSet<Value> result;

  for (auto &effect : *effects) {
    if (isa<mlir::MemoryEffects::Write>(effect.getEffect())) {
      Value val = effect.getValue();
      ASSERT_STREAM(val, "Needs to have attached value!");
      result.insert(val);
    }
  }

  return result;
}

/// Checks if it's possible to re-write the chain structure linking mutOp1 and
/// mutOp2 such that no data race is possible. This primarily comes
/// down to checking that the ops don't share an operand that has a write effect
/// on it.
static bool canSafelyReorderMutableOps(MOMutableOpInterface mutOp1,
                                       MOMutableOpInterface mutOp2) {
  // Check that the ops don't share an operand they both have a write effect on.
  auto r1 = llvm::make_filter_range(mutOp1->getOperands(), isMutableOperand);
  auto r2 = llvm::make_filter_range(mutOp2->getOperands(), isMutableOperand);

  auto mutableOperands1 = llvm::SmallDenseSet<Value>{r1.begin(), r1.end()};
  auto mutableOperands2 = llvm::SmallDenseSet<Value>{r2.begin(), r2.end()};

  auto writeEffectOperands1 = collectWriteEffectOperands(mutOp1);
  auto writeEffectOperands2 = collectWriteEffectOperands(mutOp2);

  if (llvm::set_intersects(mutableOperands1, writeEffectOperands2) ||
      llvm::set_intersects(mutableOperands2, writeEffectOperands1))
    return false;

  // It's my understanding that a kernel could potentially return a ref to an
  // input opaque type back out. In lieu of a more sophisticated alias analysis
  // (which will eventually be necessary) we simply check that there are no
  // opaque values that name the same type.
  llvm::SmallDenseSet<StringAttr> symbolSet;
  for (Value v : r1) {
    if (auto opaqueType = dyn_cast<MO::OpaqueType>(v.getType()))
      symbolSet.insert(opaqueType.getSymbol());
  }

  bool hasSameOpaqueType = llvm::any_of(r2, [&](Value v) {
    if (auto opaqueType = dyn_cast<MO::OpaqueType>(v.getType()))
      return symbolSet.contains(opaqueType.getSymbol());
    return false;
  });

  return !hasSameOpaqueType;
}

namespace {

struct CanonicalizeBufferOpSequences
    : public mlir::OpInterfaceRewritePattern<MOMutableOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult
  matchAndRewrite(MOMutableOpInterface mutOp,
                  mlir::PatternRewriter &origRewriter) const override {
    DeviceAwareRewriter rewriter(mutOp, origRewriter);

    std::optional<ChainValue> inChain = mutOp.getUniqueInChain();
    std::optional<ChainValue> outChain = mutOp.getUniqueOutChain();
    if (!inChain || !outChain)
      return rewriter.notifyMatchFailure(
          mutOp,
          "Mutable op has multiple chains; canonicalization not supported");

    rewriter.setInsertionPointAfter(mutOp);

    auto rawParent = inChain->getDefiningOp();

    // Don't rewrite chains that cross region boundaries.  An op inside a
    // region body (e.g. mo.parallel) may capture a chain from an outer-scope
    // op; bypassing that chain would break sequencing guarantees.
    if (rawParent && rawParent->getParentRegion() != mutOp->getParentRegion())
      return rewriter.notifyMatchFailure(mutOp,
                                         "chain crosses a region boundary");

    auto parent = dyn_cast_if_present<MOMutableOpInterface>(rawParent);

    if (parent && hasSingleChain(parent) &&
        canSafelyReorderMutableOps(mutOp, parent)) {
      auto parentInChain = parent.getInChains()[0];

      rewriter.modifyOpInPlace(mutOp.getOperation(), [&]() {
        mutOp.getUniqueInChainMutable()->set(parentInChain);
      });

      Value parentOutChain = parent.getOutChains()[0];

      auto merge = rewriter.create<ChainCreateOp>(
          mutOp->getLoc(), ValueRange({*outChain, parentOutChain}));

      rewriter.replaceAllUsesExcept(*outChain, merge.getResult(), merge);

      return success();
    } else if (auto chainCreateOp =
                   dyn_cast_if_present<ChainCreateOp>(rawParent)) {
      mlir::BitVector eraseIdxs(chainCreateOp->getNumOperands());

      SmallVector<Value> newCreateOperands;
      SmallVector<Value> residualOperands;

      for (auto &&[idx, operand] :
           llvm::enumerate(chainCreateOp->getOperands())) {
        if (!isa<ChainType>(operand.getType())) {
          residualOperands.push_back(operand);
          continue;
        }

        auto upstreamOp = operand.getDefiningOp<MOMutableOpInterface>();
        if (upstreamOp && hasSingleChain(upstreamOp) &&
            canSafelyReorderMutableOps(mutOp, upstreamOp)) {
          eraseIdxs[idx] = true;
          newCreateOperands.push_back(operand);
          llvm::append_range(residualOperands, upstreamOp.getInChains());
          continue;
        }

        residualOperands.push_back(operand);
      }

      if (newCreateOperands.empty())
        return rewriter.notifyMatchFailure(
            mutOp,
            "Upstream mo.chain.create has no chain operands derived from "
            "other in-place operations on non-aliased buffers");

      newCreateOperands.push_back(*outChain);

      auto merge =
          rewriter.create<ChainCreateOp>(mutOp->getLoc(), newCreateOperands);

      rewriter.replaceAllUsesExcept(*outChain, merge.getResult(), merge);

      auto newInChain =
          rewriter.create<ChainCreateOp>(mutOp->getLoc(), residualOperands);
      mutOp.getInChainsMutable()[0]->set(newInChain);
      return success();
    } else {
      return rewriter.notifyMatchFailure(
          mutOp, "The op upstream is not a chain.create or buffer mutating "
                 "operation.");
    }
  }
};

struct CanonicalizeChainSequencesPass
    : public impl::CanonicalizeChainSequencesBase<
          CanonicalizeChainSequencesPass> {
  using CanonicalizeChainSequencesBase::CanonicalizeChainSequencesBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns{ctx};

    patterns.add<CanonicalizeBufferOpSequences>(ctx);
    ChainCreateOp::getCanonicalizationPatterns(patterns, ctx);

    // Configure the greedy rewrite driver for potentially long chains.
    mlir::GreedyRewriteConfig config;
    config.setMaxIterations(750).setUseTopDownTraversal();

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      mlir::emitError(getOperation()->getLoc(),
                      "maximum number of iterations reached in "
                      "canonicalize-chain-sequences");
      return signalPassFailure();
    }
  }
};

} // namespace
