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
      regionOp.getIsKGENAttr(),
      regionOp.getSummaryAttr(),           // summary
      regionOp.getIsAsyncAttr(),           // isAsync
      regionOp.getConstantResourcesAttr(), // constantResources
      regionOp.getCanFuseIntoOtherAttr(),  // canFuseIntoOther
      regionOp.getCanBeFusedIntoAttr(),    // canBeFusedInto
      regionOp.getCanDuplicateAttr()       // canDuplicate
  );
  newKernel.setDeviceRef(regionOp.getDeviceRef());
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
      auto chainOutRange =
          llvm::make_filter_range(outputOp->getOpOperands(), [](OpOperand &op) {
            return isa<MEF::ChainType>(op.get().getType());
          });
      SmallVector<OpOperand *> chainOutVec =
          llvm::map_to_vector(chainOutRange, [](OpOperand &op) { return &op; });

      // no duplication, skip
      if (chainOutVec.size() <= 1)
        return;

      // We should keep the last chain, which is also the chain that is created
      // closet to the exit point of the region.
      OpOperand *toKeep = chainOutVec.back();
      SmallVector<Value> chainsToRemove;
      llvm::BitVector toRemoveBits(outputOp->getNumOperands());
      for (OpOperand *toRemove : ArrayRef(chainOutVec).drop_back()) {
        toRemoveBits.set(toRemove->getOperandNumber());
        chainsToRemove.push_back(toRemove->get());
      }
      // remove operands from mogg.output.
      outputOp->eraseOperands(toRemoveBits);

      for (Value toRemove : chainsToRemove) {
        // Also erase creation of the chain too (they have already been inserted
        // when the pass runs after post-abi fusion).
        for (Operation *user :
             llvm::make_early_inc_range(toRemove.getUsers())) {
          if (auto asyncAlloc = dyn_cast<MOGG::AllocateAsyncRefOp>(user)) {
            // Now that the chain is not going to escape, we no longer need to
            // create an async value for it.
            assert(asyncAlloc->hasOneUse() &&
                   isa<MOGG::CallOp>(*asyncAlloc->getUsers().begin()) &&
                   cast<MOGG::CallOp>(*asyncAlloc->getUsers().begin())
                           .getMojoKernelTargetAttr()
                           .str() == "builtin.create_chain_async");
            asyncAlloc->getUsers().begin()->erase();
            asyncAlloc.erase();
          }
        }
      }

      // Create a new region op but with collapsed result type.
      MOGG::GenericCompiledRegionOp newKernel = buildRegionOpWithResultType(
          builder, regionOp, outputOp->getOperandTypes());

      // The kept chain position relative to the new kernel results.
      unsigned pos = toKeep->getOperandNumber() - chainsToRemove.size();
      Value keptChain = newKernel->getResult(pos);
      unsigned curIdx = 0;
      for (OpResult result : regionOp->getOpResults()) {
        if (toRemoveBits.test(result.getResultNumber())) {
          result.replaceAllUsesWith(keptChain);
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
        if (operandToFirstArg.contains(operand.get())) {
          // Replace all uses of argument, then remove it from the body. Store
          // the OpOperand to remove after the fact
          arg.replaceAllUsesWith(operandToFirstArg[operand.get()]);
          regionOp.getBody().eraseArgument(operand.getOperandNumber());
          operandsToRemove.set(operand.getOperandNumber());
        } else {
          operandToFirstArg[operand.get()] = arg;
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
