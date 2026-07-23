//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Fuses connected GenericCompiledRegion.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Interfaces/LowerableToMGPInterface.h"
#include "GraphCompiler/MOGGDialect/Analysis/SequenceFusionAnalysis.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Support/AssertStream.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SetVector.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_SEQUENCEFUSION
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

namespace M::MOGG {

namespace {

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

struct SequenceFusion : public MOGG::impl::SequenceFusionBase<SequenceFusion> {
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

} // namespace M::MOGG
