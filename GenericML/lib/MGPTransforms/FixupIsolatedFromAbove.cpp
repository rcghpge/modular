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

  // Transfer existing block arguments.
  for (BlockArgument inArg : inBlock->getArguments()) {
    BlockArgument outArg =
        outBlock->addArgument(inArg.getType(), inArg.getLoc());
    mapping.map(inArg, outArg);
  }
  // Add arguments for each free var.
  for (Value inFreeValue : freeValues) {
    BlockArgument outArg =
        outBlock->addArgument(inFreeValue.getType(), inFreeValue.getLoc());
    mapping.map(inFreeValue, outArg);
  }

  for (Operation &inOp : inBlock->getOperations()) {
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
        builder.create<MEF::OutputOp>(inOp.getLoc(), outOperands);
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

      // Capture mappings for op results. Any additional results,
      // corresponding to the subOutputValues, can be ignored.
      for (const auto [inResult, outResult] :
           llvm::zip(inCfOp->getResults(), outCfOp->getResults()))
        mapping.map(inResult, outResult);

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
    builder.clone(inOp, mapping);
  }
}

ModelOp M::MGP::fixupIsolatedFromAbove(mlir::OpBuilder &builder,
                                       ModelOp inModelOp) {
  builder.setInsertionPointAfter(inModelOp);
  auto outModelOp = builder.create<MGP::ModelOp>(
      inModelOp.getLoc(),
      /*frameworkName=*/inModelOp.getFrameworkNameAttr(),
      /*inputOutputTypes=*/inModelOp.getInputOutputTypes(),
      /*sym_name=*/inModelOp.getSymName(),
      /*theExtraAttrs=*/inModelOp.getTheExtraAttrsAttr());
  fixup(builder, inModelOp.getInitBlock(), /*freeValues=*/{},
        &outModelOp.getInit().emplaceBlock(),
        /*mustAlsoReturn=*/false, /*beforeLastOutput=*/false);
  fixup(builder, inModelOp.getExecuteBlock(), /*freeValues=*/{},
        &outModelOp.getExecute().emplaceBlock(),
        /*mustAlsoReturn=*/false, /*beforeLastOutput=*/false);
  return outModelOp;
}
