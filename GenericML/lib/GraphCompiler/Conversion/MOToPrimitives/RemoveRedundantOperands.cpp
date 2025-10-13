//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Fuses connected GenericCompiledRegion.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"

#include "GraphRT/GRTDialect/GRTOps.h"

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_REMOVEREDUNDANTOPERANDS
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

inline static MOGG::GenericCompiledRegionOp
buildRegionOpWithResultType(OpBuilder &builder,
                            MOGG::GenericCompiledRegionOp regionOp,
                            TypeRange resultTypes) {
  //  Replace the RegionOp with a new RegionOp, but with deduplicated result.
  auto newKernel = builder.create<MOGG::GenericCompiledRegionOp>(
      regionOp->getLoc(), resultTypes, regionOp.getOperands(),
      regionOp.getOutputParamDeclsAttr(), regionOp.getInOutSignatureAttr(),
      regionOp.getSummaryAttr(),           // summary
      regionOp.getIsAsyncAttr(),           // isAsync
      regionOp.getConstantResourcesAttr(), // constantResources
      regionOp.getCanFuseIntoOtherAttr(),  // canFuseIntoOther
      regionOp.getCanBeFusedIntoAttr(),    // canBeFusedInto
      regionOp.getCanDuplicateAttr(),      // canDuplicate
      regionOp.getEmitsMojoAttr()          // emitsMojo
  );
  newKernel.setDeviceRef(regionOp.getExecutionDevice());
  return newKernel;
}

namespace M::MGP {

namespace {

// This pass will remove any redundant operands within a
// generic-compiled region and collapse the usage.  This reduces the number of
// async values crossing the generic-region boundary
struct RemoveRedundantOperands
    : public MGP::impl::RemoveRedundantOperandsBase<RemoveRedundantOperands> {
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
          curIdx++;
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
  }
};

} // namespace

} // namespace M::MGP
