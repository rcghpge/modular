//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MODevices.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOUtils.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "Support/AssertStream.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace M::MO {

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_CANONICALIZECHAINSEQUENCES
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

} // namespace M::MO

using namespace M;
using namespace M::MO;

static bool hasSingleChain(MOMutableOpInterface mutOp) {
  if (auto multi =
          dyn_cast<MOMultiChainMutableOpInterface>(mutOp.getOperation())) {
    if (multi.getAllInChains().size() != 1)
      return false;

    if (multi.getAllOutChains().size() != 1)
      return false;
  }
  return true;
}

static mlir::DenseSet<Value>
collectWriteEffectOperands(MOMutableOpInterface mutOp) {
  SmallVector<mlir::MemoryEffects::EffectInstance> effects;
  mutOp.getEffects(effects);

  llvm::DenseSet<Value> result;

  for (auto &&effect : effects) {
    if (isa<mlir::MemoryEffects::Write>(effect.getEffect())) {
      Value val = effect.getValue();
      // NOTE: Alternately you could assume write effects on any operands that
      // aren't encountered in this loop.
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

  auto mutableOperands1 = llvm::DenseSet<Value>{r1.begin(), r1.end()};
  auto mutableOperands2 = llvm::DenseSet<Value>{r2.begin(), r2.end()};

  auto writeEffectOperands1 = collectWriteEffectOperands(mutOp1);
  auto writeEffectOperands2 = collectWriteEffectOperands(mutOp2);

  if (llvm::any_of(mutableOperands1, [&writeEffectOperands2](Value val) {
        return writeEffectOperands2.contains(val);
      }))
    return false;

  if (llvm::any_of(mutableOperands2, [&writeEffectOperands1](Value val) {
        return writeEffectOperands1.contains(val);
      }))
    return false;

  // It's my understanding that a kernel could potentially return a ref to an
  // input opaque type back out. In lieu of a more sophisticated alias analysis
  // (which will eventually be necessary) we simply check that there are no
  // opaque values that name the same type.
  llvm::DenseSet<StringAttr> symbolSet;
  for (Value v : r1) {
    if (auto opaqueType = dyn_cast<MO::OpaqueType>(v.getType()))
      symbolSet.insert(opaqueType.getSymbol());
  }

  bool hasSameOpaqueType = llvm::any_of(r2, [&](Value v) {
    if (auto opaqueType = dyn_cast<MO::OpaqueType>(v.getType()))
      return symbolSet.count(opaqueType.getSymbol()) > 0;
    return false;
  });

  return !hasSameOpaqueType;
}

namespace {

struct CanonicalizeBufferOpSequences : public mlir::RewritePattern {

  CanonicalizeBufferOpSequences(MLIRContext *ctx)
      : RewritePattern(mlir::Pattern::MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op,
                  mlir::PatternRewriter &origRewriter) const override {
    DeviceAwareRewriter rewriter(op, origRewriter);

    auto mutOp = dyn_cast<MOMutableOpInterface>(op);
    if (!mutOp) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), "Expected op to be derived from MOMutableOpInterface");
    }

    if (!hasSingleChain(mutOp)) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          "Mutable op has multiple chains; canonicalization not supported");
    }

    auto inChain = mutOp.getInChain();
    auto outChain = mutOp.getOutChain();

    rewriter.setInsertionPointAfter(mutOp);

    auto rawParent = inChain.getDefiningOp();

    auto parent = dyn_cast_if_present<MOMutableOpInterface>(rawParent);

    if (parent && !hasSingleChain(parent))
      parent = nullptr;

    if (parent && canSafelyReorderMutableOps(mutOp, parent)) {
      mutOp.getInChainMutable().set(parent.getInChain());

      Value parentOutChain = parent.getOutChain();

      auto merge = rewriter.create<ChainCreateOp>(
          mutOp->getLoc(), ValueRange({outChain, parentOutChain}));

      rewriter.replaceAllUsesExcept(outChain, merge.getResult(), merge);

      return success();
    } else if (auto chainCreateOp =
                   dyn_cast_if_present<ChainCreateOp>(rawParent)) {
      mlir::BitVector eraseIdxs(chainCreateOp->getNumOperands());

      SmallVector<Value> newCreateOperands;

      for (auto &&[idx, operand] :
           llvm::enumerate(chainCreateOp->getOperands())) {
        if (!isa<ChainType>(operand.getType()))
          continue;

        auto upstreamOp = operand.getDefiningOp<MOMutableOpInterface>();
        if (upstreamOp && !hasSingleChain(upstreamOp))
          upstreamOp = nullptr;
        if (upstreamOp && canSafelyReorderMutableOps(mutOp, upstreamOp)) {
          eraseIdxs[idx] = true;
          newCreateOperands.push_back(operand);
        }
      }

      if (newCreateOperands.empty()) {
        return rewriter.notifyMatchFailure(
            mutOp,
            "Upstream mo.chain.create has no chain operands derived from "
            "other in-place operations on non-aliased buffers");
      }

      newCreateOperands.push_back(outChain);

      auto merge =
          rewriter.create<ChainCreateOp>(mutOp->getLoc(), newCreateOperands);

      rewriter.replaceAllUsesExcept(outChain, merge.getResult(), merge);

      chainCreateOp->eraseOperands(eraseIdxs);
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

    // Configure the greedy rewrite driver for potentially long chains.
    mlir::GreedyRewriteConfig config;
    config.setMaxIterations(128).setUseTopDownTraversal();

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
