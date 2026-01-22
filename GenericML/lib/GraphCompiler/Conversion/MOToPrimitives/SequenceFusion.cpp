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
#include "GenericML/GraphCompiler/Interfaces/LowerableToMGPInterface.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/SequenceFusionAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SetVector.h"

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_SEQUENCEFUSION
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

namespace M::MGP {

namespace {

void fuseAndReplaceCompiledRegions(bool allowDuplication,
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
    // properly. The duplicated chain will be removed later in the pipeline
    // (namely, remove-redundant-operands pass).
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
  // For now we set the device to be a cuda device if one exists.
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
      /*summary=*/
      MOGG::getSummaryString(toFuse, MOGG::FusionType::SOFT),
      /*isAsync=*/BoolAttr::get(ctx, isAsync),
      /*constantResources=*/builder.getArrayAttr(resources.getArrayRef()),
      canFuseIntoOther, canBeFusedInto, canDuplicate, isStructuredKernel);
  fusedCompiledRegionOp.setDeviceRef(device);
  Block &newBody = fusedCompiledRegionOp.getBody().emplaceBlock();
  builder.setInsertionPoint(&newBody, newBody.begin());

  // Create a mapping from outside SSA value to internal SSA value for inputs.
  DenseMap<Value, Value> outsideValueToInternalValue;
  for (Value operand : fusedRegionOperands) {
    auto newArg = newBody.addArgument(operand.getType(), loc);
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
      // Actual fusion.
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

  // Creating the output op and ensure that there is at most one async ready op
  // in the kernel.
  auto newOutputOp = MOGG::OutputOp::create(builder, loc, valuesToOutput);
  if (!chainsThatShouldBeRemapped.empty()) {
    auto firstReadyOp = *newBody.getOps<MOGG::AsyncReadyOp>().begin();
    firstReadyOp->moveBefore(newOutputOp);
    newOutputOp->insertOperands(0, firstReadyOp.getResult());

    for (auto &&chain : chainsThatShouldBeRemapped)
      chain.replaceAllUsesWith(fusedCompiledRegionOp.getResult(0));
  }

  // Use-def replacement.
  for (auto [oldValue, newValue] :
       llvm::zip(resultsThatShouldBeReplaced.takeVector(),
                 fusedCompiledRegionOp.getResults().drop_front(
                     !chainsThatShouldBeRemapped.empty()))) {
    oldValue.replaceAllUsesWith(newValue);
  }
}

/// TODO(GEX-2986): Remove this function.
static SmallVector<SmallVector<Operation *>>
splitIfTooLarge(ArrayRef<Operation *> compiledRegionOpsToFuseTogether) {
  SmallVector<SmallVector<Operation *>> fusionSets;
  static constexpr uint64_t kMaxKernelsInFusionSet = 16;
  uint64_t setSize = kMaxKernelsInFusionSet;
  for (auto &&op : compiledRegionOpsToFuseTogether) {
    if (setSize == kMaxKernelsInFusionSet) {
      fusionSets.emplace_back();
      setSize = 0;
    }

    fusionSets.back().push_back(op);
    setSize += cast<MOGG::GenericCompiledRegionOp>(op).getIsStructuredKernel();
  }

  return fusionSets;
}

// This function will sequence fuse a set of MOGG::GenericCompiledRegionOp. It
// expects the input vector to be topologically sorted.
template <bool allowDuplication>
static void fuseCompiledRegionsAndReplaceUses(
    ArrayRef<Operation *> compiledRegionOpsToFuseTogether) {
  auto fusionSets = splitIfTooLarge(compiledRegionOpsToFuseTogether);
  for (auto &&fusionSet : fusionSets)
    fuseAndReplaceCompiledRegions(allowDuplication, fusionSet);
}

static inline bool isaGenericCompiledRegionOp(Operation *op) {
  return isa<MOGG::GenericCompiledRegionOp>(op);
}

static inline bool isFusibleIntoOther(Operation *op) {
  if (auto asRegion = dyn_cast<MOGG::GenericCompiledRegionOp>(op))
    return asRegion.getCanFuseIntoOther();

  return false;
}

static llvm::SetVector<MGP::DeviceContextType>
getUsedContexts(MOGG::GenericCompiledRegionOp kernel) {
  llvm::SetVector<MGP::DeviceContextType> devices;
  for (auto operandTy : kernel.getOperandTypes()) {
    if (auto ctx = dyn_cast<MGP::DeviceContextType>(operandTy))
      devices.insert(ctx);
  }

  return devices;
}

/// Kernels that may have an identical execution device can still use input
/// buffers that are located on different devices (ex:
/// mgp.buffer.device_to_device). These kernels should not be sequence fused
/// (see GEX-2590).
static bool useSameContexts(MOGG::GenericCompiledRegionOp lhs,
                            MOGG::GenericCompiledRegionOp rhs) {
  auto lhsContexts = getUsedContexts(lhs);
  auto rhsContexts = getUsedContexts(rhs);
  if (lhsContexts.empty() || rhsContexts.empty())
    return true;

  return lhsContexts == rhsContexts;
}

// checkForDuplication controls if we should check duplication. If set, we check
// that duplicationFlag is the same as getCanDuplicate for the parent op.
template <bool checkForDuplication, bool duplicationFlag = false>
static inline bool areCompatibleForFusion(Operation *parent, Operation *child) {
  auto parentRegion = dyn_cast<MOGG::GenericCompiledRegionOp>(parent);
  auto childRegion = dyn_cast<MOGG::GenericCompiledRegionOp>(child);

  if (!parentRegion || !childRegion)
    return false;

  if (!parentRegion.getCanFuseIntoOther())
    return false;

  if (!childRegion.getCanBeFusedInto())
    return false;

  if (!useSameContexts(parentRegion, childRegion))
    return false;

  if constexpr (checkForDuplication) {
    if (duplicationFlag != parentRegion.getCanDuplicate())
      return false;
  }

  return true;
}

static inline bool isComputationallySignificant(Operation *op) {
  if (auto asRegion = dyn_cast<MOGG::GenericCompiledRegionOp>(op))
    return !asRegion.getCanFuseIntoOther();
  return false;
}

// SourceDestination sequence fusion is the simplest form of fusion: fuse things
// down into a region until nothing else can be fused. allowDuplication controls
// if we allow src ops to duplicate into many dst ops.
template <bool allowDuplication>
static void doSourceDestinationSequenceFusion(MGP::ModelOp model) {
  auto canDuplicate = [](Operation *op) -> bool { return allowDuplication; };
  // After fusion, the ParametricTopologyAnalysis won't change because there are
  // no parameters in MGP.
  MO::ParametricTopologyAnalysis parametricTopologyAnalysis(model);
  for (auto &region : model->getRegions()) {
    auto subgraphs = MOGG::sourceDestinationSequenceFusionAnalysis(
        region, parametricTopologyAnalysis, isaGenericCompiledRegionOp,
        areCompatibleForFusion<true, allowDuplication>, canDuplicate);

    for (auto &subgraph : subgraphs)
      parametricTopologyAnalysis.computeTopologicalSorting(subgraph);

    mlir::SetVector<Operation *> opsToErase;
    for (auto &subgraph : subgraphs)
      opsToErase.insert(subgraph.begin(), subgraph.end());
    SmallVector<Operation *> opsToEraseOrdered(opsToErase.begin(),
                                               opsToErase.end());
    parametricTopologyAnalysis.computeTopologicalSorting(opsToEraseOrdered);

    for (auto &subgraph : subgraphs)
      fuseCompiledRegionsAndReplaceUses<allowDuplication>(subgraph);

    for (auto op : llvm::reverse(opsToEraseOrdered))
      op->erase();
  }
}

// This kind of fusion will fuse things that are not necessarily connected via
// graph edges. It produces kernels with multiple outputs. It is configured to
// maintain parallelism of computationally expensive operations.
[[maybe_unused]] static void
doTransitiveDependencySequenceFusion(MGP::ModelOp model) {
  // After fusion, the ParametricTopologyAnalysis won't change because there are
  // no parameters in MGP.
  MO::ParametricTopologyAnalysis parametricTopologyAnalysis(model);
  for (auto &region : model->getRegions()) {
    auto subgraphs = MOGG::transitiveDependencySequenceFusionAnalysis(
        region, parametricTopologyAnalysis, isFusibleIntoOther,
        areCompatibleForFusion<false>, isComputationallySignificant, true);

    for (auto &subgraph : subgraphs)
      parametricTopologyAnalysis.computeTopologicalSorting(subgraph);

    mlir::SetVector<Operation *> opsToErase;
    for (auto &subgraph : subgraphs)
      opsToErase.insert(subgraph.begin(), subgraph.end());
    SmallVector<Operation *> opsToEraseOrdered(opsToErase.begin(),
                                               opsToErase.end());
    parametricTopologyAnalysis.computeTopologicalSorting(opsToEraseOrdered);
    for (auto &subgraph : subgraphs)
      fuseCompiledRegionsAndReplaceUses<false>(subgraph);

    for (auto op : llvm::reverse(opsToEraseOrdered))
      op->erase();
  }
}

struct SequenceFusion : public MGP::impl::SequenceFusionBase<SequenceFusion> {
  using SequenceFusionBase::SequenceFusionBase;

  void runOnOperation() override {
    MGP::ModelOp model = getOperation();

    // Fuse in any op that can be duplicated.
    doSourceDestinationSequenceFusion<true>(model);

    // Fuse in any op that can't be duplicated (must have one user).
    doSourceDestinationSequenceFusion<false>(model);

    if (enableTransitiveFusion)
      doTransitiveDependencySequenceFusion(model);
  }
};

} // namespace

} // namespace M::MGP
