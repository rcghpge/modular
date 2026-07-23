//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Implementation of fuseAndReplaceCompiledRegions.
//
// This lives in its own translation unit (rather than in FusionUtils.cpp or
// SequenceFusion.cpp) so it can be linked by both GraphCompilerInterfaces and
// MOGGTransforms without introducing a circular Bazel dependency.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"

#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Support/AssertStream.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SetVector.h"

using namespace M;

namespace M::MOGG {

GenericCompiledRegionOp
fuseAndReplaceCompiledRegions(bool allowDuplication,
                              ArrayRef<Operation *> toFuse) {
  ASSERT_STREAM(llvm::all_of(toFuse,
                             [](auto op) {
                               return isa<MOGG::GenericCompiledRegionOp>(op);
                             }),
                << "Expected all operations to be "
                << MOGG::GenericCompiledRegionOp::getOperationName());
  // Iterate through all the ops to fuse and gather all operand and results.
  llvm::SetVector<Value> allResults;
  llvm::SetVector<Value> allOperands;
  for (Operation *op : toFuse) {
    auto results = op->getResults();
    auto operands = op->getOperands();
    allResults.insert(results.begin(), results.end());
    allOperands.insert(operands.begin(), operands.end());
  }

  // The fused op's operands will be allOperands - allResults.
  llvm::SetVector<Value> fusedRegionOperands = allOperands;
  fusedRegionOperands.set_subtract(allResults);

  // Iterate through all the results and identify values that need to be return
  // values after fusion.
  llvm::SetVector<Value> resultsThatShouldBeReplaced;
  SmallVector<Type> fusedRegionOutputTypes;
  DenseSet<Operation *> compiledRegionOpsToFuseTogetherAsSet(toFuse.begin(),
                                                             toFuse.end());

  SmallVector<Value> chainsThatShouldBeRemapped;
  for (auto result : allResults) {
    // We keep at least one chain output such that error could be propagated
    // properly. Duplicated chain outputs and operands are removed later in the
    // pipeline by the GenericCompiledRegionOp canonicalization.
    bool isChain = isa<MEF::ChainType>(result.getType());
    if (isChain) {
      chainsThatShouldBeRemapped.push_back(result);
      continue;
    }

    if (allowDuplication) {
      // If none of the users of this result are in the fusion set, then it is
      // something that is solely used from the outside.
      if (llvm::none_of(result.getUsers(), [&](auto user) {
            return compiledRegionOpsToFuseTogetherAsSet.contains(user);
          })) {
        resultsThatShouldBeReplaced.insert(result);
        fusedRegionOutputTypes.push_back(result.getType());
      }
    } else {
      // If any of the users of this result are used outside, then it must be
      // output too.
      if (llvm::any_of(result.getUsers(), [&](auto user) {
            return !compiledRegionOpsToFuseTogetherAsSet.contains(user);
          })) {
        resultsThatShouldBeReplaced.insert(result);
        fusedRegionOutputTypes.push_back(result.getType());
      }
    }
  }

  if (!chainsThatShouldBeRemapped.empty()) {
    fusedRegionOutputTypes.insert(fusedRegionOutputTypes.begin(),
                                  chainsThatShouldBeRemapped.front().getType());
  }

  // Maintain attributes.
  auto firstRegionOp = cast<MOGG::GenericCompiledRegionOp>(toFuse.front());
  const bool isAsync = llvm::any_of(toFuse, [&](auto region) {
    return cast<MOGG::GenericCompiledRegionOp>(region).getIsAsync().value_or(
        false);
  });
  llvm::SmallSetVector<Attribute, 4> resources;
  for (Operation *regionOp : toFuse) {
    auto res = cast<MOGG::GenericCompiledRegionOp>(regionOp)
                   .getConstantResourcesAttr()
                   .getValue();
    resources.insert(res.begin(), res.end());
  }
  const bool canFuseIntoOther = llvm::all_of(toFuse, [&](auto region) {
    return cast<MOGG::GenericCompiledRegionOp>(region).getCanFuseIntoOther();
  });
  const bool canBeFusedInto = llvm::all_of(toFuse, [&](auto region) {
    return cast<MOGG::GenericCompiledRegionOp>(region).getCanBeFusedInto();
  });
  const bool canDuplicate = llvm::all_of(toFuse, [&](auto region) {
    return cast<MOGG::GenericCompiledRegionOp>(region).getCanDuplicate();
  });
  const bool isStructuredKernel = llvm::any_of(toFuse, [&](auto region) {
    return cast<MOGG::GenericCompiledRegionOp>(region).getIsStructuredKernel();
  });
  DeviceRefAttr device =
      cast<MOGG::GenericCompiledRegionOp>(toFuse.front()).getExecutionDevice();
  for (auto region : toFuse) {
    auto regionDevice =
        cast<MOGG::GenericCompiledRegionOp>(region).getExecutionDevice();
    if (regionDevice.getLabel() == kGPULabel)
      device = regionDevice;
  }

  auto ctx = firstRegionOp->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPointAfter(toFuse.back());

  SmallVector<Location> locs;
  for (auto region : toFuse)
    locs.push_back(region->getLoc());
  auto loc = mlir::FusedLoc::get(ctx, locs);

  SmallVector<InOutSignatureAttr::InOutSemantics> operandSemantics;
  auto fusedCompiledRegionOp = MOGG::GenericCompiledRegionOp::create(
      builder, loc, fusedRegionOutputTypes, fusedRegionOperands.getArrayRef(),
      KGEN::ParamDeclArrayAttr::get(ctx, {}),
      InOutSignatureAttr::get(ctx, operandSemantics),
      /*fusionInfo=*/
      MOGG::createFusionInfo<MOGG::FusionSequenceAttr>(ctx, toFuse),
      /*isAsync=*/BoolAttr::get(ctx, isAsync),
      /*constantResources=*/builder.getArrayAttr(resources.getArrayRef()),
      canFuseIntoOther, canBeFusedInto, canDuplicate, isStructuredKernel);
  fusedCompiledRegionOp.setDeviceRef(device);
  Block &newBody = fusedCompiledRegionOp.getBody().emplaceBlock();
  builder.setInsertionPoint(&newBody, newBody.begin());

  // Create a mapping from outside SSA value to internal SSA value for inputs.
  // Body block arguments are `!mogg.async<T>` wrappers around the operand
  // types.
  DenseMap<Value, Value> outsideValueToInternalValue;
  for (Value operand : fusedRegionOperands) {
    auto newArg =
        newBody.addArgument(MOGG::AsyncType::get(operand.getType()), loc);
    outsideValueToInternalValue[operand] = newArg;
  }

  SmallVector<Value> valuesToOutput;

  IRMapping mapper;
  for (auto op : toFuse) {
    auto asRegionOp = cast<MOGG::GenericCompiledRegionOp>(op);
    for (auto [operand, argument] :
         llvm::zip(asRegionOp->getOperands(),
                   asRegionOp.getBody().front().getArguments())) {
      mapper.map(argument, outsideValueToInternalValue[operand]);
    }

    for (auto &internalOp : asRegionOp.getBody().front()) {
      auto newOp = builder.clone(internalOp, mapper);
      auto newOutputOp = dyn_cast<MOGG::OutputOp>(newOp);
      if (!newOutputOp)
        continue;

      for (auto [oldResult, internalValue] :
           llvm::zip(asRegionOp->getResults(), newOutputOp.getOperands())) {
        outsideValueToInternalValue[oldResult] = internalValue;
        if (resultsThatShouldBeReplaced.contains(oldResult))
          valuesToOutput.push_back(internalValue);
      }

      newOutputOp->erase();
    }
  }

  auto newOutputOp = MOGG::OutputOp::create(builder, loc, valuesToOutput);
  if (!chainsThatShouldBeRemapped.empty()) {
    auto firstReadyOp = *newBody.getOps<MOGG::AsyncReadyOp>().begin();
    firstReadyOp->moveBefore(newOutputOp);
    newOutputOp->insertOperands(0, firstReadyOp.getResult());

    for (auto &&chain : chainsThatShouldBeRemapped)
      chain.replaceAllUsesWith(fusedCompiledRegionOp.getResult(0));
  }

  for (auto [oldValue, newValue] :
       llvm::zip(resultsThatShouldBeReplaced.takeVector(),
                 fusedCompiledRegionOp.getResults().drop_front(
                     !chainsThatShouldBeRemapped.empty()))) {
    oldValue.replaceAllUsesWith(newValue);
  }

  return fusedCompiledRegionOp;
}

} // namespace M::MOGG
