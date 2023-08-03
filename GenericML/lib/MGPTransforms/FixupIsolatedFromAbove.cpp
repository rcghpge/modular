//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "./FixupIsolatedFromAbove.h"

#include "GraphRT/GRTDialect/GRTControlFlowInterface.h"
#include "GraphRT/MEFDialect/MEFOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"

#include "./Utils.h"

#define DEBUG_TYPE "mgp-fixup-isolated-from-above"

using namespace M;
using namespace M::MGP;
using namespace mlir;

/// Accumulates all the values in block (and its sub-blocks) which are bound
/// outside of bindingBlock.
static void accumFreeValues(SetVector<Value> &outerValues, Block *block,
                            Block *bindingBlock) {
  for (Operation &inOp : block->getOperations()) {
    for (Value inValue : inOp.getOperands()) {
      if (inValue.getParentBlock() != bindingBlock)
        outerValues.insert(inValue);
    }
    for (Region &subRegion : inOp.getRegions())
      accumFreeValues(outerValues, &subRegion.front(), bindingBlock);
  }
}

[[maybe_unused]] static SetVector<Value> getValues(Operation *op) {
  SetVector<Value> result;
  for (Value value : op->getResults())
    result.insert(value);
  for (Value value : op->getOperands())
    result.insert(value);
  return result;
}

/// Returns false if a value is shared between inOp and outOp.
/// The caller is expected to then assert fail.
static bool checkRewrite(Operation *inOp, Operation *outOp) {
#if MODULAR_DEBUG
  SetVector<Value> inValues = getValues(inOp);
  SetVector<Value> outValues = getValues(outOp);
  SetVector<Value> intersection = llvm::set_intersection(inValues, outValues);
  if (!intersection.empty()) {
    llvm::errs() << "a rewritten operation is continuing to use values shared "
                    "with the original operation.\n";
    llvm::errs() << "input operation:\n" << *inOp << "\n";
    llvm::errs() << "output operation:\n" << *outOp << "\n";
    for (Value v : intersection)
      llvm::errs() << "shared value:\n" << v << "\n";
    return false;
  }
#endif
  return true;
}

/// Rewrites inBlock into outBlock, where freeValues will instead be bound
/// by new arguments in outBlock. If mustAlsoReturn is true, freeValues will
/// also become new block outputs. If, in addition, beforeLastOutput is true,
/// any new block outputs will come just before the existing last block output.
///
/// Could be reused at higher dialects provided they implement
/// GRTControlFlowInterface for the control flow ops, and we don't depend
/// on the concrete MEF::OutputOp type for checking for output terminators.
static void fixup(OpBuilder &builder, Block *inBlock,
                  const SetVector<Value> &freeValues, Block *outBlock,
                  bool mustAlsoReturn, bool beforeLastOutput) {
  IRMapping mapping;
  builder.setInsertionPoint(outBlock, outBlock->begin());

  LLVM_DEBUG(llvm::dbgs() << "fixup begin\n");

  // Transfer existing block arguments.
  for (BlockArgument inArg : inBlock->getArguments()) {
    BlockArgument outArg =
        outBlock->addArgument(inArg.getType(), inArg.getLoc());
    LLVM_DEBUG(llvm::dbgs() << "mapping " << inArg << " to " << outArg << "\n");
    mapping.map(inArg, outArg);
  }
  // Add arguments for each free var.
  for (Value inFreeValue : freeValues) {
    BlockArgument outArg =
        outBlock->addArgument(inFreeValue.getType(), inFreeValue.getLoc());
    LLVM_DEBUG(llvm::dbgs()
               << "mapping " << inFreeValue << " to " << outArg << "\n");
    mapping.map(inFreeValue, outArg);
  }

  for (Operation *inOp : getDataflowOrder(inBlock)) {
    LLVM_DEBUG(llvm::dbgs() << "looking at op " << *inOp << "\n");
    if (auto inOutputOp = dyn_cast<MEF::OutputOp>(inOp)) {
      if (mustAlsoReturn) {
        SmallVector<Value> outOperands;
        // Rewrite existing operands (possibly excluding last).
        size_t n = inOutputOp->getNumOperands();
        if (beforeLastOutput) {
          assert(n > 0);
          --n;
        }
        for (size_t i = 0; i < n; ++i)
          outOperands.emplace_back(mapping.lookup(inOutputOp->getOperand(i)));
        // Include the additional arguments.
        for (Value inFreeValue : freeValues)
          outOperands.emplace_back(mapping.lookup(inFreeValue));
        if (beforeLastOutput)
          // Rewrite last operand.
          outOperands.emplace_back(mapping.lookup(inOutputOp->getOperand(n)));
        Operation *outOutputOp =
            builder.create<MEF::OutputOp>(inOp->getLoc(), outOperands);
        assert(checkRewrite(inOutputOp, outOutputOp) &&
               "invalid rewrite for output op");
        continue;
      }
    }

    if (auto inCfOp = dyn_cast<GRT::GRTControlFlowInterface>(inOp)) {
      bool subMustAlsoReturn = inCfOp.getFlavor() == GRT::kLoopLike;

      // Collect the free values for the control flow op sub-blocks.
      SetVector<Value> subFreeValues;
      for (Region &region : inCfOp->getRegions())
        accumFreeValues(subFreeValues, &region.front(), &region.front());
      assert(llvm::all_of(subFreeValues, [&](Value subFreeValue) {
        return subFreeValue.getParentBlock() == inBlock ||
               freeValues.contains(subFreeValue);
      }));

      // Rewrite existing operands.
      SmallVector<Value> outOperands;
      for (Value operand : inCfOp->getOperands())
        outOperands.emplace_back(mapping.lookup(operand));
      // Include the additional arguments.
      for (Value inSubFreeValue : subFreeValues)
        outOperands.emplace_back(mapping.lookup(inSubFreeValue));

      // New op continues to return existing results (possibly excluding last).
      size_t n = inCfOp->getNumResults();
      if (beforeLastOutput) {
        assert(n > 0);
        --n;
      }
      SmallVector<Type> outResultTypes;
      for (size_t i = 0; i < n; ++i)
        outResultTypes.emplace_back(inCfOp->getResult(i).getType());
      if (subMustAlsoReturn) {
        // But new op must also return a result for each new argument.
        for (Value inSubFreeValue : subFreeValues)
          outResultTypes.emplace_back(inSubFreeValue.getType());
      }
      if (beforeLastOutput)
        // Account for last result.
        outResultTypes.emplace_back(inCfOp->getResult(n).getType());

      // Create the new control flow op with empty sub-regions.
      GRT::GRTControlFlowInterface outCfOp =
          cloneWithArgs(builder, inCfOp, outResultTypes, outOperands);
      assert(checkRewrite(inCfOp, outCfOp) &&
             "invalid rewrite for control flow op");

      // Capture mappings for op results. Any additional results,
      // corresponding to the subOutputValues, can be ignored.
      for (const auto [inResult, outResult] :
           llvm::zip(inCfOp->getResults(), outCfOp->getResults())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "mapping " << inResult << " to " << outResult << "\n");
        mapping.map(inResult, outResult);
      }

      // Rewrite the sub-blocks.
      auto insertPoint = builder.saveInsertionPoint();
      for (size_t i = 0, m = inCfOp->getNumRegions(); i < m; ++i) {
        // The 'cond' sub-block for a grt.while must always return the
        // condition as the last argument. So any new values must go before
        // that.
        bool subBeforeLastOutput = inCfOp.getFlavor() == GRT::kLoopLike &&
                                   inCfOp->getNumRegions() == 2 && i == 0;
        Block *inSubBlock = &inCfOp->getRegion(i).front();
        Block *outSubBlock = &outCfOp->getRegion(i).emplaceBlock();
        fixup(builder, inSubBlock, subFreeValues, outSubBlock,
              subMustAlsoReturn, subBeforeLastOutput);
      }
      builder.restoreInsertionPoint(insertPoint);
      continue;
    }

    // Default case is just clone the op w.r.t. mapping.
    Operation *outOp = builder.clone(*inOp, mapping);
    assert(checkRewrite(inOp, outOp) && "invalid rewrite of op");
  }

  LLVM_DEBUG(llvm::dbgs() << "fixup end\n");
}

ModelOp M::MGP::fixupIsolatedFromAbove(mlir::OpBuilder &builder,
                                       ModelOp inModelOp) {
  LLVM_DEBUG({
    llvm::dbgs() << "before fixupIsolatedFromAbove:\n";
    inModelOp.print(llvm::dbgs(), mlir::OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n\n";
  });

  builder.setInsertionPointAfter(inModelOp);
  auto outModelOp = builder.create<MGP::ModelOp>(
      inModelOp.getLoc(),
      /*frameworkName=*/inModelOp.getFrameworkNameAttr(),
      /*inputOutputTypes=*/inModelOp.getInputOutputTypes(),
      /*sym_name=*/inModelOp.getSymName(),
      /*theExtraAttrs=*/inModelOp.getTheExtraAttrsAttr());
  LLVM_DEBUG(llvm::dbgs() << "init\n");
  fixup(builder, inModelOp.getInitBlock(),
        /*freeValues=*/{}, &outModelOp.getInit().emplaceBlock(),
        /*mustAlsoReturn=*/false, /*beforeLastOutput=*/false);
  LLVM_DEBUG(llvm::dbgs() << "execute\n");
  fixup(builder, inModelOp.getExecuteBlock(),
        /*freeValues=*/{}, &outModelOp.getExecute().emplaceBlock(),
        /*mustAlsoReturn=*/false, /*beforeLastOutput=*/false);
  return outModelOp;
}
