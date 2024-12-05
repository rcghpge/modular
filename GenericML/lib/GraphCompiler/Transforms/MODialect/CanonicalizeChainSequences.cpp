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

static bool opsHaveAliasedMutableOperands(Operation *op1, Operation *op2) {
  auto isMutableOperand = [](Value val) {
    return isa<MO::BufferType, MO::OpaqueType>(val.getType());
  };

  auto r1 = llvm::make_filter_range(op1->getOperands(), isMutableOperand);
  auto r2 = llvm::make_filter_range(op2->getOperands(), isMutableOperand);

  auto s1 = mlir::DenseSet<Value>{r1.begin(), r1.end()};
  auto s2 = mlir::DenseSet<Value>{r2.begin(), r2.end()};

  return llvm::any_of(s1, [&s2](Value val) { return s2.contains(val); });
}

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

    auto inChain = mutOp.getInChain();
    auto outChain = mutOp.getOutChain();

    rewriter.setInsertionPointAfter(mutOp);

    auto rawParent = inChain.getDefiningOp();

    auto parent = llvm::dyn_cast_or_null<MOMutableOpInterface>(rawParent);

    if (parent && !opsHaveAliasedMutableOperands(mutOp, parent)) {
      mutOp.getInChainMutable().set(parent.getInChain());

      Value parentOutChain = parent.getOutChain();

      auto merge = rewriter.create<ChainCreateOp>(
          mutOp->getLoc(), ValueRange({outChain, parentOutChain}));

      rewriter.replaceAllUsesExcept(outChain, merge.getResult(), merge);

      return success();
    } else if (auto chainCreateOp =
                   dyn_cast_or_null<ChainCreateOp>(rawParent)) {
      mlir::BitVector eraseIdxs(chainCreateOp->getNumOperands());

      SmallVector<Value> newCreateOperands;

      for (auto &&[idx, operand] :
           llvm::enumerate(chainCreateOp->getOperands())) {
        if (!isa<ChainType>(operand.getType()))
          continue;

        auto upstreamOp = operand.getDefiningOp();
        if (isa<MOMutableOpInterface>(upstreamOp) &&
            !opsHaveAliasedMutableOperands(mutOp, upstreamOp)) {
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

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
