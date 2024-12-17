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
    ArrayRef<Operation *> compiledRegionOpsToFuseTogether) {
  MOGG::fuseAndReplaceCompiledRegions(allowDuplication,
                                      compiledRegionOpsToFuseTogether);
}

// FIXME(GEX-462) remove stop-gap once MOCO-6 is done.
// Heuristic from empirical observations. Empirically saw a stack overflow at
// around 200 outputs with nested if-else, set the heuristic to 100 for
// buffering room since this heuristic may not translate directly to the levels
// of nesting in kgen.
constexpr size_t kMaxAssertOpsFused = 100;

// FIXME(GEX-462) remove stop-gap once MOCO-6 is done.
// splitSubGraphFusionByNumOps allows us to limit the maximum number of
// operations in a CompiledRegion.  We opted for two simple counting heuristics
// instead of mapping to kgen since this is meant to be a temporary measure.
// Stop fusing of regions rules:
//   1. mgp.assertions (which cause nesting) exceeds 150
static SmallVector<SmallVector<Operation *>> splitSubGraphFusionByNumOps(
    const SmallVector<Operation *> &compiledRegionOpsToFuseTogether) {
  assert(llvm::all_of(compiledRegionOpsToFuseTogether, [](auto op) {
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

  if (!childRegion.getCanBeFusedInto())
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
      // TODO(GEX-784): NOTE: all generic regions should not allow duplication.
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
