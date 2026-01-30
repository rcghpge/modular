//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Fuses connected GenericCompiledRegion.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Support/AssertStream.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_REMOVEREDUNDANTOPERANDS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

inline static MOGG::GenericCompiledRegionOp
buildRegionOpWithResultType(OpBuilder &builder,
                            MOGG::GenericCompiledRegionOp regionOp,
                            TypeRange resultTypes) {
  //  Replace the RegionOp with a new RegionOp, but with deduplicated result.
  auto newKernel = MOGG::GenericCompiledRegionOp::create(
      builder, regionOp->getLoc(), resultTypes, regionOp.getOperands(),
      regionOp.getOutputParamDeclsAttr(), regionOp.getInOutSignatureAttr(),
      regionOp.getSummaryAttr(),           // summary
      regionOp.getIsAsyncAttr(),           // isAsync
      regionOp.getConstantResourcesAttr(), // constantResources
      regionOp.getCanFuseIntoOtherAttr(),  // canFuseIntoOther
      regionOp.getCanBeFusedIntoAttr(),    // canBeFusedInto
      regionOp.getCanDuplicateAttr()       // canDuplicate
  );
  newKernel.setDeviceRef(regionOp.getExecutionDevice());
  return newKernel;
}

namespace M::MOGG {

namespace {

// Pattern to remove unused results from TryOp and update yield operations
struct RemoveUnusedTryResults : public OpRewritePattern<MOGG::TryOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MOGG::TryOp tryOp,
                                PatternRewriter &rewriter) const override {

    // Collect which results are actually used
    SmallVector<unsigned> usedResultIndices;
    SmallVector<Type> usedResultTypes;
    for (auto [idx, result] : llvm::enumerate(tryOp.getResults())) {
      if (!result.use_empty()) {
        usedResultIndices.push_back(idx);
        usedResultTypes.push_back(result.getType());
      }
    }

    // If all results are used, no change needed
    if (usedResultIndices.size() == tryOp.getNumResults())
      return failure();

    // Create a new try op with only the used result types
    auto newTryOp = MOGG::TryOp::create(rewriter, tryOp.getLoc(),
                                        TypeRange(usedResultTypes));

    // Clone the try region
    rewriter.cloneRegionBefore(tryOp.getTryRegion(), newTryOp.getTryRegion(),
                               newTryOp.getTryRegion().begin());

    // Clone the except region
    rewriter.cloneRegionBefore(tryOp.getExceptRegion(),
                               newTryOp.getExceptRegion(),
                               newTryOp.getExceptRegion().begin());

    // Update the yield operation in the try region
    auto tryYield =
        cast<MOGG::YieldOp>(newTryOp.getTryRegion().front().getTerminator());
    SmallVector<Value> newTryYieldOperands;
    for (unsigned idx : usedResultIndices) {
      newTryYieldOperands.push_back(tryYield.getOperand(idx));
    }
    rewriter.setInsertionPoint(tryYield);
    rewriter.replaceOpWithNewOp<MOGG::YieldOp>(tryYield, newTryYieldOperands);

    // Update the yield operation in the except region
    auto exceptYield =
        cast<MOGG::YieldOp>(newTryOp.getExceptRegion().front().getTerminator());
    SmallVector<Value> newExceptYieldOperands;
    SmallVector<Value> usedExceptValues;
    for (unsigned idx : usedResultIndices) {
      Value operand = exceptYield.getOperand(idx);
      newExceptYieldOperands.push_back(operand);
      usedExceptValues.push_back(operand);
    }
    rewriter.setInsertionPoint(exceptYield);
    rewriter.replaceOpWithNewOp<MOGG::YieldOp>(exceptYield,
                                               newExceptYieldOperands);

    // Erase unused AsyncErrorOp operations from the except block to prevent CSE
    // from merging them. Each AsyncErrorOp must remain distinct because they
    // correspond to specific output indices for emit-mojo code generation.
    SmallVector<Operation *> opsToErase;
    for (Operation &op : newTryOp.getExceptRegion().front()) {
      if (auto asyncError = dyn_cast<MOGG::AsyncErrorOp>(&op)) {
        if (!llvm::is_contained(usedExceptValues, asyncError.getResult())) {
          opsToErase.push_back(&op);
        }
      }
    }
    for (Operation *op : opsToErase) {
      rewriter.eraseOp(op);
    }

    // Replace uses of the old try op with the new one
    // Map old results to new results for used indices
    SmallVector<Value> replacements(tryOp.getNumResults());
    for (auto [newIdx, oldIdx] : llvm::enumerate(usedResultIndices)) {
      replacements[oldIdx] = newTryOp.getResult(newIdx);
    }

    rewriter.replaceOp(tryOp, replacements);
    return success();
  }
};

// This pass will remove any redundant operands within a
// generic-compiled region and collapse the usage.  This reduces the number of
// async values crossing the generic-region boundary
struct RemoveRedundantOperands
    : public MOGG::impl::RemoveRedundantOperandsBase<RemoveRedundantOperands> {
  using RemoveRedundantOperandsBase::RemoveRedundantOperandsBase;

  void runOnOperation() override {
    auto modelOp = getOperation();
    auto ctx = modelOp->getContext();
    OpBuilder builder(ctx);

    // Run two passes over the module:
    // 1st pass removes duplicate !mef.chain from output.
    modelOp->walk([&](MOGG::GenericCompiledRegionOp regionOp) {
      builder.setInsertionPoint(regionOp);
      auto outputOp =
          cast<MOGG::OutputOp>(regionOp.getRegion().front().getTerminator());
      // Filter out duplicated !mef.chain output.
      auto unUsedOrChain = llvm::make_filter_range(
          outputOp->getOpOperands(), [&](OpOperand &op) {
            return isa<MEF::ChainType>(op.get().getType()) ||
                   regionOp->getResult(op.getOperandNumber()).use_empty();
          });
      SmallVector<OpOperand *> unUsedOrChainVec =
          llvm::map_to_vector(unUsedOrChain, [](OpOperand &op) { return &op; });

      // no chain output, skip
      if (unUsedOrChainVec.empty())
        return;

      llvm::BitVector toRemoveBits(outputOp->getNumOperands());
      bool usedChain = false;
      for (OpOperand *toRemove : unUsedOrChainVec) {
        unsigned idx = toRemove->getOperandNumber();

        // Check whether all chains has no use.
        if (isa<MEF::ChainType>(toRemove->get().getType()))
          usedChain = usedChain || !regionOp->getResult(idx).use_empty();

        toRemoveBits.set(idx);
      }

      OpOperand *keptChain = nullptr;
      if (usedChain || toRemoveBits.all()) {
        // We keep the last chain if
        // 1. any of the chain has uses or
        // 2. the chain is the only output after folding.
        for (OpOperand &op : llvm::reverse(outputOp->getOpOperands())) {
          if (isa<MEF::ChainType>(op.get().getType())) {
            toRemoveBits.reset(op.getOperandNumber());
            keptChain = &op;
            break;
          }
        }
      }

      // Can we simply erase the regionOp in this case?
      ASSERT_STREAM(
          !toRemoveBits.all(),
          "A region should at least have one used result or it should have "
          "a chain result for side effect:\n"
              << regionOp);

      // No need to deduplicate
      if (toRemoveBits.none())
        return;

      SmallVector<Value> resultsToRemove = llvm::map_to_vector(
          toRemoveBits.set_bits(), [outputOp](unsigned operandNumber) {
            return outputOp->getOperand(operandNumber);
          });

      // remove operands from mogg.output.
      outputOp->eraseOperands(toRemoveBits);

      for (Value toRemove : resultsToRemove) {
        if (!isa<MEF::ChainType>(toRemove.getType())) {
          ASSERT_STREAM(toRemove.use_empty(), << toRemove << "has uses");
          continue;
        }
        // Also erase creation of the chain too (they have already been inserted
        // when the pass runs after post-abi fusion).
        for (Operation *user :
             llvm::make_early_inc_range(toRemove.getUsers())) {
          if (auto asyncAlloc = dyn_cast<MOGG::PackToAsyncRef>(user)) {
            for (Operation *op :
                 llvm::make_early_inc_range(asyncAlloc->getUsers())) {
              // Must be a deallocation, since we no longer need the chain, they
              // can be erased too.
              assert(isa<MOGG::DeallocateAsyncRefOp>(op));
              op->erase();
            }
            asyncAlloc.erase();
          }
        }
        // If this is simple result of a chain.create operation, erase it.
        auto chainCreate = toRemove.getDefiningOp<MOGG::CreateReadyChainOp>();
        if (chainCreate)
          chainCreate->erase();
      }

      // Create a new region op but with collapsed result type.
      MOGG::GenericCompiledRegionOp newKernel = buildRegionOpWithResultType(
          builder, regionOp, outputOp->getOperandTypes());

      // The kept chain position relative to the new kernel results.
      unsigned curIdx = 0;
      for (OpResult result : regionOp->getOpResults()) {
        if (toRemoveBits.test(result.getResultNumber())) {
          if (keptChain && isa<MEF::ChainType>(result.getType())) {
            unsigned pos =
                keptChain->getOperandNumber() - resultsToRemove.size();
            result.replaceAllUsesWith(newKernel->getResult(pos));
          }
        } else {
          result.replaceAllUsesWith(newKernel.getResult(curIdx));
          ++curIdx;
        }
      }
      assert(curIdx == newKernel->getNumResults());
      regionOp.getBody().front().moveBefore(&newKernel.getBody(),
                                            newKernel.getBody().end());
      regionOp->erase();
      return;
    });

    // 2nd pass removes duplicate inputs (this includes duplicated !mef.chain
    // input introduced in the 1st pass above).
    modelOp.walk([&](MOGG::GenericCompiledRegionOp regionOp) {
      SmallDenseMap<Value, Value> operandToFirstArg;
      llvm::BitVector operandsToRemove(regionOp.getNumOperands());
      // Collapse all OpOperands that are duplicates and collapse all arguments
      // within region to use the same one, go in reverse order
      for (auto &operand : llvm::reverse(regionOp->getOpOperands())) {
        auto arg =
            regionOp.getBody().front().getArgument(operand.getOperandNumber());
        auto [it, inserted] = operandToFirstArg.try_emplace(operand.get(), arg);
        if (!inserted) {
          // Replace all uses of argument, then remove it from the body. Store
          // the OpOperand to remove after the fact
          arg.replaceAllUsesWith(it->second);
          regionOp.getBody().eraseArgument(operand.getOperandNumber());
          operandsToRemove.set(operand.getOperandNumber());
        }
      }

      regionOp->eraseOperands(operandsToRemove);

      regionOp.setInOutSignatureAttr(
          regionOp.getInOutSignature().remove(operandsToRemove));
    });

    // Apply pattern rewrites for TryOp after all redundant operand removal
    RewritePatternSet patterns(ctx);
    patterns.add<RemoveUnusedTryResults>(ctx);

    if (failed(applyPatternsGreedily(modelOp, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

} // namespace M::MOGG
