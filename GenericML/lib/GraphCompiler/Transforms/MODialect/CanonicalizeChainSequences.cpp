//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MODevices.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
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

namespace {

static bool opsHaveAliasedBuffers(Operation *op1, Operation *op2) {
  auto isBuffer = [](Value val) { return isa<MO::BufferType>(val.getType()); };

  auto r1 = llvm::make_filter_range(op1->getOperands(), isBuffer);
  auto r2 = llvm::make_filter_range(op2->getOperands(), isBuffer);

  auto s1 = mlir::DenseSet<Value>{r1.begin(), r1.end()};
  auto s2 = mlir::DenseSet<Value>{r2.begin(), r2.end()};

  return llvm::any_of(s1, [&s2](Value val) { return s2.contains(val); });
}

static bool isInPlaceCustomOp(CustomOp customOp) {
  return customOp.getBufferOperands().size() >= 1 &&
         customOp.maybeGetInChain() && customOp.maybeGetOutChain();
}

// TODO(GRA-1078): Ultimately what we probably really want is an interface for
// ops that are can represent in-place compute (mo.custom and mogg.kernel atm)
// that aren't the core mutable ops.
static bool isSideEffectingBufferOp(Operation *op) {
  if (isa<MO::MOMutableOpInterface>(op))
    return true;
  else if (auto customOp = dyn_cast<CustomOp>(op))
    return isInPlaceCustomOp(customOp);
  return false;
}

static ChainValue getInChain(Operation *op) {
  if (auto mutOp = dyn_cast<MOMutableOpInterface>(op)) {
    return mutOp.getInChain();
  } else if (auto customOp = dyn_cast<CustomOp>(op)) {
    return customOp.maybeGetInChain().value_or(ChainValue{});
  }
  llvm_unreachable("getInChain is only valid on side-effecting buffer ops");
}

static ChainValue getOutChain(Operation *op) {
  if (auto mutOp = dyn_cast<MOMutableOpInterface>(op)) {
    return mutOp.getOutChain();
  } else if (auto customOp = dyn_cast<CustomOp>(op)) {
    return customOp.maybeGetOutChain().value_or(ChainValue{});
  }
  llvm_unreachable("getOutChain is only valid on side-effecting buffer ops");
}

static size_t getInChainIdx(Operation *op, Value chain) {
  for (auto [idx, value] : llvm::enumerate(op->getOperands())) {
    if (value == chain) {
      return idx;
    }
  }
  llvm_unreachable("This function should only be called when the input chain "
                   "is known to be one of op's operands.");
}

struct CanonicalizeBufferOpSequences : public mlir::RewritePattern {

  CanonicalizeBufferOpSequences(MLIRContext *ctx)
      : RewritePattern(mlir::Pattern::MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op,
                  mlir::PatternRewriter &origRewriter) const override {
    DeviceAwareRewriter rewriter(op, origRewriter);

    if (!isSideEffectingBufferOp(op)) {
      return rewriter.notifyMatchFailure(
          op, "Not an op that has side-effects on a buffer, skipping.");
    }

    auto inChain = getInChain(op);
    auto outChain = getOutChain(op);

    rewriter.setInsertionPointAfter(op);

    auto parent = inChain.getDefiningOp();

    if (isSideEffectingBufferOp(parent) && !opsHaveAliasedBuffers(op, parent)) {
      auto idx = getInChainIdx(op, inChain);

      rewriter.rewriter.startOpModification(op);
      op->getOpOperand(idx).set(getInChain(parent));
      rewriter.rewriter.finalizeOpModification(op);

      auto merge = rewriter.create<ChainCreateOp>(
          op->getLoc(), ValueRange({outChain, getOutChain(parent)}));

      rewriter.replaceAllUsesExcept(outChain, merge.getResult(), merge);

      return success();
    } else if (auto chainCreateOp = dyn_cast_or_null<ChainCreateOp>(parent)) {
      mlir::BitVector eraseIdxs(chainCreateOp->getNumOperands());

      SmallVector<Value> newCreateOperands;

      for (auto &&[idx, operand] :
           llvm::enumerate(chainCreateOp->getOperands())) {
        if (!isa<ChainType>(operand.getType()))
          continue;

        auto upstreamOp = operand.getDefiningOp();
        if (isSideEffectingBufferOp(upstreamOp)) {
          if (!opsHaveAliasedBuffers(op, upstreamOp)) {
            eraseIdxs[idx] = true;
            newCreateOperands.push_back(operand);
          }
        }
      }

      if (newCreateOperands.empty()) {
        return rewriter.notifyMatchFailure(
            op, "Upstream mo.chain.create has no chain operands derived from "
                "other in-place operations on non-aliased buffers");
      }

      newCreateOperands.push_back(outChain);

      auto merge =
          rewriter.create<ChainCreateOp>(op->getLoc(), newCreateOperands);

      rewriter.replaceAllUsesExcept(outChain, merge.getResult(), merge);

      chainCreateOp->eraseOperands(eraseIdxs);
      return success();
    } else {
      return rewriter.notifyMatchFailure(
          op, "The op upstream is not a chain.create or buffer mutating "
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

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
