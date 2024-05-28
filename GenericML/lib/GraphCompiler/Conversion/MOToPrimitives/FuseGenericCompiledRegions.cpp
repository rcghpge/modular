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
#include "mlir/IR/IRMapping.h"

#include "GraphRT/GRTDialect/GRTOps.h"

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_FUSEGENERICCOMPILEDREGIONS
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

namespace M::MGP {

namespace {

// This function will sequence fuse a set of MOGG::GenericCompiledRegionOp. It
// expects the input vector to be topologically sorted.
template <bool allowDuplication>
static void fuseCompiledRegionsAndReplaceUses(
    const SmallVector<Operation *> &compiledRegionOpsToFuseTogether) {
  ASSERT_STREAM(llvm::all_of(compiledRegionOpsToFuseTogether, [](auto op) {
    return isa<MOGG::GenericCompiledRegionOp>(op);
  }));

  // Iterate through all the ops to fuse and gather all operand and results.
  llvm::SetVector<Value> allResults;
  llvm::SetVector<Value> allOperands;
  for (Operation *op : compiledRegionOpsToFuseTogether) {
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
  DenseSet<Operation *> compiledRegionOpsToFuseTogetherAsSet(
      compiledRegionOpsToFuseTogether.begin(),
      compiledRegionOpsToFuseTogether.end());
  for (auto result : allResults) {
    if constexpr (allowDuplication) {
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

  // Maintain attributes.
  auto firstRegionOp = cast<MOGG::GenericCompiledRegionOp>(
      compiledRegionOpsToFuseTogether.front());
  const bool isAsync =
      llvm::any_of(compiledRegionOpsToFuseTogether, [&](auto region) {
        return cast<MOGG::GenericCompiledRegionOp>(region)
            .getIsAsync()
            .value_or(false);
      });
  const bool isFusible =
      llvm::all_of(compiledRegionOpsToFuseTogether, [&](auto region) {
        return cast<MOGG::GenericCompiledRegionOp>(region)
            .getCanFuseIntoOther();
      });
  const bool canDuplicate =
      llvm::all_of(compiledRegionOpsToFuseTogether, [&](auto region) {
        return cast<MOGG::GenericCompiledRegionOp>(region).getCanDuplicate();
      });
  ASSERT_STREAM(llvm::all_of(compiledRegionOpsToFuseTogether, [&](auto region) {
    return cast<MOGG::GenericCompiledRegionOp>(region).getIsKGEN().value_or(
               false) == firstRegionOp.getIsKGEN().value_or(false);
  }));

  auto ctx = firstRegionOp->getContext();
  OpBuilder builder(ctx);
  SmallVector<Location> locs;
  for (auto region : compiledRegionOpsToFuseTogether)
    locs.push_back(region->getLoc());
  auto loc = mlir::FusedLoc::get(ctx, locs);
  builder.setInsertionPointAfter(compiledRegionOpsToFuseTogether.back());
  SmallVector<InOutSignatureAttr::InOutSemantics> operandSemantics;
  auto fusedCompiledRegionOp = builder.create<MOGG::GenericCompiledRegionOp>(
      loc, fusedRegionOutputTypes, fusedRegionOperands.getArrayRef(),
      KGEN::ParamDeclArrayAttr::get(ctx, {}),
      InOutSignatureAttr::get(ctx, operandSemantics),
      firstRegionOp.getIsKGENAttr(),
      MOGG::getSummaryString(compiledRegionOpsToFuseTogether),
      BoolAttr::get(ctx, isAsync), isFusible, canDuplicate);
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
  for (auto op : compiledRegionOpsToFuseTogether) {
    auto asRegionOp = cast<MOGG::GenericCompiledRegionOp>(op);
    for (auto [operand, argument] :
         llvm::zip(asRegionOp->getOperands(),
                   asRegionOp.getBody().front().getArguments())) {
      mapper.map(argument, outsideValueToInternalValue[operand]);
    }

    for (auto &internalOp : asRegionOp.getBody().front()) {
      // Actual fusion.
      auto newOp = builder.clone(internalOp, mapper);
      if (auto newOutputOp = dyn_cast<MOGG::OutputOp>(newOp)) {
        auto resultRange =
            llvm::make_filter_range(asRegionOp->getResults(), [](Value val) {
              return !isa<MEF::ChainType>(val.getType());
            });

        auto updateMetadataFunc = [&](Value oldResult, Value internalValue) {
          outsideValueToInternalValue[oldResult] = internalValue;
          if (resultsThatShouldBeReplaced.contains(oldResult))
            valuesToOutput.push_back(internalValue);
        };

        if (asRegionOp->getNumResults() != newOutputOp->getNumOperands()) {
          for (auto [oldResult, internalValue] :
               llvm::zip(resultRange, newOutputOp.getOperands()))
            updateMetadataFunc(oldResult, internalValue);
        } else {
          for (auto [oldResult, internalValue] :
               llvm::zip(asRegionOp->getResults(), newOutputOp.getOperands()))
            updateMetadataFunc(oldResult, internalValue);
        }

        newOutputOp->erase();
      }
    }
  }

  ASSERT_STREAM(valuesToOutput.size() == resultsThatShouldBeReplaced.size());

  // Creating the output op and replacing outside SSA values.
  builder.create<MOGG::OutputOp>(loc, valuesToOutput);
  for (auto [oldValue, newValue] :
       llvm::zip(resultsThatShouldBeReplaced.takeVector(),
                 fusedCompiledRegionOp.getResults())) {
    oldValue.replaceAllUsesWith(newValue);
  }
}

// FIXME(GRA-462) remove stop-gap once MOCO-6 is done.
// Heuristic from empirical observations. Empirically saw a stack overflow at
// around 200 outputs with nested if-else, set the heuristic to 100 for
// buffering room since this heuristic may not translate directly to the levels
// of nesting in kgen.
constexpr size_t kMaxAssertOpsFused = 100;

// FIXME(GRA-462) remove stop-gap once MOCO-6 is done.
// splitSubGraphFusionByNumOps allows us to limit the maximum number of
// operations in a CompiledRegion.  We opted for two simple counting heuristics
// instead of mapping to kgen since this is meant to be a temporary measure.
// Stop fusing of regions rules:
//   1. mgp.assertions (which cause nesting) exceeds 150
static SmallVector<SmallVector<Operation *>> splitSubGraphFusionByNumOps(
    const SmallVector<Operation *> &compiledRegionOpsToFuseTogether) {
  ASSERT_STREAM(llvm::all_of(compiledRegionOpsToFuseTogether, [](auto op) {
    return isa<MOGG::GenericCompiledRegionOp>(op);
  }));
  unsigned int numAssertOpsFused = 0;
  SmallVector<SmallVector<Operation *>> splitSubgraphs = {{}};
  for (auto op : compiledRegionOpsToFuseTogether) {
    unsigned int numAssertOpsInCompiledRegion = 0;
    // Count all assertion ops
    for (auto &region : op->getRegions())
      for (auto op : region.getOps<MOGG::CallOp>())
        if (op.getMojoKernelTarget() == "mgp.assert")
          numAssertOpsFused++;
    // Break or allow fusion
    if ((numAssertOpsFused + numAssertOpsInCompiledRegion) >
        kMaxAssertOpsFused) {
      splitSubgraphs.push_back({op});
      numAssertOpsFused = numAssertOpsInCompiledRegion;
    } else {
      splitSubgraphs.back().push_back(op);
      numAssertOpsFused += numAssertOpsInCompiledRegion;
    }
  }
  return splitSubgraphs;
}

static inline bool isaGenericCompiledRegionOp(Operation *op) {
  return isa<MOGG::GenericCompiledRegionOp>(op);
}

static inline bool isFusibleIntoOther(Operation *op) {
  if (auto asRegion = dyn_cast<MOGG::GenericCompiledRegionOp>(op))
    return asRegion.getCanFuseIntoOther();

  return false;
}

// checkForDuplication controls if we should check duplication. If set, we check
// that duplicationFlag is the same as getCanDuplicate for the parent op.
template <bool checkForDuplication, bool duplicationFlag = false>
static inline bool areCompatibleForFusion(Operation *parent, Operation *child) {
  auto parentRegion = dyn_cast<MOGG::GenericCompiledRegionOp>(parent);
  auto childRegion = dyn_cast<MOGG::GenericCompiledRegionOp>(child);

  if (!parentRegion || !childRegion)
    return false;

  if (parentRegion.getIsKGEN() != childRegion.getIsKGEN())
    return false;

  if (!parentRegion.getCanFuseIntoOther())
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

  return isa<MGP::KernelExecuteOp>(op);
}

// SourceDestination sequence fusion is the simplest form of fusion: fuse things
// down into a region until nothing else can be fused. allowDuplication controls
// if we allow src ops to duplicate into many dst ops.
template <bool allowDuplication>
static void doSourceDestinationSequenceFusion(ModuleOp module) {
  for (auto model : module.getOps<MGP::ModelOp>()) {
    for (auto &region : model->getRegions()) {
      MO::ParametricTopologyAnalysis parametricTopologyAnalysis(model);
      auto subgraphs = MOGG::sourceDestinationSequenceFusionAnalysis(
          region, parametricTopologyAnalysis, isaGenericCompiledRegionOp,
          areCompatibleForFusion<true, allowDuplication>, allowDuplication);

      for (auto &subgraph : subgraphs)
        parametricTopologyAnalysis.computeTopologicalSorting(subgraph);

      DenseSet<Operation *> opsToErase;
      for (auto &subgraph : subgraphs)
        opsToErase.insert(subgraph.begin(), subgraph.end());
      SmallVector<Operation *> opsToEraseOrdered(opsToErase.begin(),
                                                 opsToErase.end());
      parametricTopologyAnalysis.computeTopologicalSorting(opsToEraseOrdered);

      for (auto &subgraph : subgraphs)
        for (auto &splitSubGraph : splitSubGraphFusionByNumOps(subgraph))
          fuseCompiledRegionsAndReplaceUses<allowDuplication>(splitSubGraph);

      for (auto op : llvm::reverse(opsToEraseOrdered))
        op->erase();
    }
  }
}

// This kind of fusion will fuse things that are not necessarily connected via
// graph edges. It produces kernels with multiple outputs. It is configured to
// maintain parallelism of computationally expensive operations.
[[maybe_unused]] static void
doTransitiveDependencySequenceFusion(ModuleOp module) {
  for (auto model : module.getOps<MGP::ModelOp>()) {
    for (auto &region : model->getRegions()) {
      MO::ParametricTopologyAnalysis parametricTopologyAnalysis(model);
      auto subgraphs = MOGG::transitiveDependencySequenceFusionAnalysis(
          region, parametricTopologyAnalysis, isFusibleIntoOther,
          areCompatibleForFusion<false>, isComputationallySignificant, true);

      for (auto &subgraph : subgraphs)
        parametricTopologyAnalysis.computeTopologicalSorting(subgraph);

      DenseSet<Operation *> opsToErase;
      for (auto &subgraph : subgraphs)
        opsToErase.insert(subgraph.begin(), subgraph.end());
      SmallVector<Operation *> opsToEraseOrdered(opsToErase.begin(),
                                                 opsToErase.end());
      parametricTopologyAnalysis.computeTopologicalSorting(opsToEraseOrdered);
      for (auto &subgraph : subgraphs)
        for (auto &splitSubGraph : splitSubGraphFusionByNumOps(subgraph))
          fuseCompiledRegionsAndReplaceUses<false>(splitSubGraph);

      for (auto op : llvm::reverse(opsToEraseOrdered))
        op->erase();
    }
  }
}

struct FuseGenericCompiledRegions
    : public MGP::impl::FuseGenericCompiledRegionsBase<
          FuseGenericCompiledRegions> {
  using FuseGenericCompiledRegionsBase::FuseGenericCompiledRegionsBase;

  void runOnOperation() override {
    if (enableFusion) {
      ModuleOp module = getOperation();
      // Fuse in any op that can be duplicated.
      doSourceDestinationSequenceFusion<true>(module);
      // Fuse in any op that can't be duplicated (must have one user).
      doSourceDestinationSequenceFusion<false>(module);
      // Fuse in anything that remains while preserving parallelism.
      doTransitiveDependencySequenceFusion(module);
    }
  }
};

} // namespace

} // namespace M::MGP
