//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/MGPTransforms.h"

#include "GenericML/MGPDialect/MGPBufferLifetimeAnalysis.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPTransforms/Utils.h"
#include "GraphRT/MIPDialect/MIPDialect.h"
#include "Support/AssertStream.h"
#include <numeric>

#define DEBUG_TYPE "mgp-dynamic-memory-planning"

using namespace mlir;

namespace M::MGP {

// Generate support code for DynamicMemoryPlanningPass
#define GEN_PASS_DEF_DYNAMICMEMORYPLANNINGPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

struct AllocInfo {
  MGP::BufferAllocDynamicOp alloc;
  Value runtimeSize;
  uint64_t alignment;
  uint64_t minPre;
  uint64_t maxPre;
  uint64_t minPost;
  uint64_t maxPost;
};

// Return true if the given value is trivially computable (i.e.,
// arithmetic and tensor_spec ops) from a block argument. This is used to
// determine if a given value representing the size of an allocation is
// data-dependent
bool isTriviallyComputable(Value v, DenseSet<Value> &trivial) {
  Operation *producer = v.getDefiningOp();

  // Block argument is trivial
  if (!producer)
    return true;

  // If we already know this is trivial, continue
  if (trivial.contains(v))
    return true;

  // If the producing op isn't trivial, this isn't a trivial value
  bool isTrivialOp =
      isa<MIP::MIPDialect>(*producer->getDialect()) ||
      isa<TensorExtractTensorSpecOp, CpuTensorSpecSizeOp, CpuTensorSpecCreateOp,
          CpuTensorSpecDimsDynamicOp, BufferAllocDynamicOp>(*producer);
  if (!isTrivialOp)
    return false;

  // If all operands of the defining op aren't trivial, this isn't a trivial
  // value
  if (!llvm::all_of(producer->getOperands(), [&](Value operand) {
        return isTriviallyComputable(operand, trivial);
      }))
    return false;

  trivial.insert(v);

  return true;
}

class DynamicMemoryPlanningPass
    : public impl::DynamicMemoryPlanningPassBase<DynamicMemoryPlanningPass> {
  using DynamicMemoryPlanningPassBase::DynamicMemoryPlanningPassBase;
  using PostDominationRange = std::pair<int, int>;

  void runOnOperation() override {
    ModelOp modelOp = getOperation();

    auto analysis = getAnalysis<BufferLifetimeAnalysisPass>();

    BufferBuilderContext builderContext =
        TensorBufferDialectInterface::getAllBufferBuilders(modelOp);

    ExecuteState execState(modelOp->getContext(), modelOp);

    // Get dependency tree for the current block
    auto &tree = analysis.getDepTree(execState.execute);

    DenseMap<DeviceRefAttr, SmallVector<AllocInfo>> deviceToAllocs;

    DenseSet<Value> trivialVals;

    for (auto allocOp :
         execState.execute->getOps<MGP::BufferAllocDynamicOp>()) {
      // Ignore CPU allocations for now. Will be addressed by GRA-557
      if (allocOp.getDevice().getLabel() == "cpu")
        continue;

      auto allocUses = getAllocUses(allocOp);

      // Skip any allocations that have no uses or have untraceable aliases
      if (allocUses.isError() || allocUses->empty())
        continue;

      // Ignore data-dependent allocations
      if (!isTriviallyComputable(allocOp.getRuntimeSize(), trivialVals))
        continue;

      uint64_t maxPre = std::numeric_limits<uint64_t>::min();
      uint64_t minPre = std::numeric_limits<uint64_t>::max();
      uint64_t minPost = std::numeric_limits<uint64_t>::max();
      uint64_t maxPost = std::numeric_limits<uint64_t>::min();

      bool ineligibleAlloc = false;
      for (auto use : allocUses.get()) {
        if (!tree.contains(use)) {
          ineligibleAlloc = true;
          break;
        }
        maxPre =
            std::max(maxPre, static_cast<uint64_t>(tree.get(use)->preIndex));
        minPost =
            std::min(minPost, static_cast<uint64_t>(tree.get(use)->postIndex));
        minPre =
            std::min(minPre, static_cast<uint64_t>(tree.get(use)->preIndex));
        maxPost =
            std::max(maxPost, static_cast<uint64_t>(tree.get(use)->postIndex));
      }

      // If the tree didn't have information on every use of this alloc, skip
      // it to be safe
      if (ineligibleAlloc)
        continue;

      auto &allocs = deviceToAllocs[allocOp.getDevice()];
      allocs.push_back({allocOp, allocOp.getRuntimeSize(),
                        allocOp.getRawAlign(), minPre, maxPre, minPost,
                        maxPost});
    }

    for (auto &[device, allocs] : deviceToAllocs) {
      if (allocs.size() <= 1)
        continue;

      // Sort allocs by postorder
      llvm::sort(allocs, [&](const auto &a, const auto &b) {
        return a.minPost < b.minPost;
      });

      // Construct and populate BufferPlanOp input arrays
      SmallVector<uint64_t> alignments;
      SmallVector<uint64_t> minPres;
      SmallVector<uint64_t> minPosts;
      SmallVector<uint64_t> maxPres;
      SmallVector<uint64_t> maxPosts;
      SmallVector<Value> runtimeSizes;

      uint64_t totalAlignment = 1;
      for (const auto &a : allocs) {
        runtimeSizes.push_back(a.runtimeSize);
        alignments.push_back(a.alignment);
        minPres.push_back(a.minPre);
        minPosts.push_back(a.minPost);
        maxPres.push_back(a.maxPre);
        maxPosts.push_back(a.maxPost);
        totalAlignment = std::lcm(totalAlignment, a.alignment);
      }

      auto loc = allocs[0].alloc->getLoc();
      execState.builder.setInsertionPointToStart(modelOp.getExecuteBlock());

      auto bufferBuilder = builderContext.getBufferBuilder(device);

      // Create BufferPlanOp
      auto bufferPlanOp = execState.builder.create<BufferPlanOp>(
          loc, runtimeSizes, alignments, minPres, minPosts, maxPres, maxPosts);

      // Create the "super-alloc" with the execute block's root chain as the
      // gating chain.
      auto superAlloc = bufferBuilder->bufferAllocDynamic(
          execState.builder, loc, execState.getContext(),
          execState.getRootChain(), bufferPlanOp.getHighWatermark(),
          totalAlignment);

      // Replace all allocations with slices into the super alloc.
      // Since all planned runtimeSizes are not data-dependent, this shouldn't
      // introduce any graph cycles.
      for (size_t i = 0; i < allocs.size(); ++i) {
        auto slice = execState.builder.create<BufferSliceOp>(
            loc, superAlloc, bufferPlanOp.getOffsets()[i],
            allocs[i].runtimeSize);
        allocs[i].alloc.getOutBuffer().replaceAllUsesWith(slice);
      }

      for (size_t i = 0; i < allocs.size(); ++i)
        allocs[i].alloc.erase();
    }
  }
};
} // namespace M::MGP
