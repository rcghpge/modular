//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/MGPTransforms.h"

#include "GenericML/MGPDialect/MGPBufferLifetimeAnalysis.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPTransforms/Utils.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "Support/AssertStream.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <numeric>

#define DEBUG_TYPE "mgp-dynamic-memory-planning"

using namespace mlir;

namespace M::MGP {

// Generate support code for DynamicMemoryPlanningPass
#define GEN_PASS_DEF_DYNAMICMEMORYPLANNINGPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

namespace {

struct AllocInfo {
  MGP::BufferAllocOp alloc;
  Value runtimeSize;
  uint64_t alignment;
  uint64_t minPre;
  uint64_t maxPre;
  uint64_t minPost;
  uint64_t maxPost;
};

/// Return true if the given value is trivially computable (i.e.,
/// arithmetic and tensor_spec ops) from a block argument. This is used to
/// determine if a given value representing the size of an allocation is
/// data-dependent.
///
/// The results of the analysis are cached on a per-operation basis in the
/// `trivial` cache. The 'isTrivial' property is determined on a per-operation
/// basis, so we cache the results as a mapping from operations to bools.
bool isTriviallyComputable(Value v, DenseMap<Operation *, bool> &trivial) {
  Operation *producer = v.getDefiningOp();

  // Block argument is trivial
  if (!producer)
    return true;

  // If we already know this is trivial, continue
  if (auto it = trivial.find(producer); it != trivial.end())
    return it->second;

  auto cacheResult = [&](bool result) -> bool {
    trivial[producer] = result;
    return result;
  };

  // If the producing op isn't trivial, this isn't a trivial value
  bool isTrivialOp = isa<KGEN::POP::POPDialect>(*producer->getDialect()) ||
                     isa<mlir::index::IndexDialect>(*producer->getDialect()) ||
                     isa<TensorExtractTensorSpecOp, TensorSpecCreateOp,
                         BufferAllocOp, TensorSpecGetDimOp>(*producer);
  if (!isTrivialOp)
    return cacheResult(false);

  // If all operands of the defining op aren't trivial, this isn't a trivial
  // value
  auto trivialOperands =
      llvm::all_of(producer->getOperands(), [&](Value operand) {
        return isTriviallyComputable(operand, trivial);
      });

  return cacheResult(trivialOperands);
}

} // end namespace

class DynamicMemoryPlanningPass
    : public impl::DynamicMemoryPlanningPassBase<DynamicMemoryPlanningPass> {
  using Base::Base;

  void runOnOperation() override {
    ModelOp modelOp = getOperation();

    auto getDeviceContext = [&modelOp](DeviceRefAttr dev,
                                       ArrayRef<BlockArgument> devices) {
      for (auto [deviceRefAttr, deviceContext] :
           llvm::zip_equal(modelOp.getDevices(), devices))
        if (deviceRefAttr == dev)
          return std::optional<Value>(deviceContext);
      return std::optional<Value>();
    };

    auto &analysis = getAnalysis<BufferLifetimeAnalysisPass>();

    BufferBuilderContext builderContext =
        TensorBufferDialectInterface::getAllBufferBuilders(modelOp);

    ExecuteState execState(modelOp->getContext(), modelOp);
    OpBuilder &execBuilder = execState.builder;

    // Get dependency tree for the current block
    DependencyTree &tree = analysis.getDepTree(execState.execute);

    llvm::MapVector<DeviceRefAttr, SmallVector<AllocInfo>> deviceToAllocs;

    DenseMap<Operation *, bool> trivialVals;

    for (auto allocOp : execState.execute->getOps<MGP::BufferAllocOp>()) {
      // Ignore CPU allocations for now. Will be addressed by GEX-557
      if (allocOp.getDevice().getLabel() == kCPULabel)
        continue;

      ErrorOr<SmallVector<Operation *>> allocUses = getAllocUses(allocOp);

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
        DependencyTreeNode *node = tree.get(use);
        if (!node) {
          ineligibleAlloc = true;
          break;
        }
        maxPre = std::max(maxPre, static_cast<uint64_t>(node->preIndex));
        minPost = std::min(minPost, static_cast<uint64_t>(node->postIndex));
        minPre = std::min(minPre, static_cast<uint64_t>(node->preIndex));
        maxPost = std::max(maxPost, static_cast<uint64_t>(node->postIndex));
      }

      // If the tree didn't have information on every use of this alloc, skip
      // it to be safe
      if (ineligibleAlloc)
        continue;

      SmallVector<AllocInfo> &allocs = deviceToAllocs[allocOp.getDevice()];
      allocs.push_back({allocOp, allocOp.getRuntimeSize(),
                        allocOp.getRawAlign(), minPre, maxPre, minPost,
                        maxPost});
    }

    // Set insertion point before the new chain because it makes the IR easier
    // to read.
    execBuilder.setInsertionPointToStart(modelOp.getExecuteBlock());

    SmallVector<BufferAllocOp> allocsToDelete;
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

      // Keep track of the LCM of all alignments so we can use this alignment
      // for the main allocation
      uint64_t totalAlignment = 1;
      for (const AllocInfo &a : allocs) {
        runtimeSizes.push_back(a.runtimeSize);
        alignments.push_back(a.alignment);
        minPres.push_back(a.minPre);
        minPosts.push_back(a.minPost);
        maxPres.push_back(a.maxPre);
        maxPosts.push_back(a.maxPost);
        totalAlignment = std::lcm(totalAlignment, a.alignment);
      }

      auto locations = llvm::map_to_vector(
          allocs, [](const AllocInfo &alloc) { return alloc.alloc->getLoc(); });

      auto loc = execBuilder.getFusedLoc(locations);
      auto bufferBuilder = builderContext.getBufferBuilder(device);

      std::optional<Value> deviceContext =
          getDeviceContext(device, modelOp.getExecuteDevices());
      ASSERT_STREAM(deviceContext, "Could not find device context for device");

      // Create BufferPlanOp in execute block
      auto bufferPlanOp =
          BufferPlanOp::create(execBuilder, loc, runtimeSizes, alignments,
                               minPres, minPosts, maxPres, maxPosts);

      TypedValue<IndexType> poolSize = bufferPlanOp.getHighWatermark();

      // Allocate a larger buffer.
      Value alloc = bufferBuilder->bufferAlloc(
          execBuilder, loc, execState.getContext(), execState.getRootChain(),
          poolSize, deviceContext.value(), totalAlignment);

      auto memPoolBuffer = cast<TypedValue<DeviceBufferType>>(alloc);

      // Replace all allocations with slices into the memPoolBuffer.
      // Since all planned runtimeSizes are not data-dependent, this
      // shouldn't introduce any graph cycles.
      for (size_t i = 0; i < allocs.size(); ++i) {
        Value slice = execBuilder.create<BufferSliceOp>(
            loc, memPoolBuffer, bufferPlanOp.getOffsets()[i],
            allocs[i].runtimeSize);

        allocs[i].alloc->replaceAllUsesWith(ValueRange{slice});
        allocsToDelete.push_back(allocs[i].alloc);
      }
    }

    // Finally delete all the old allocations.
    for (BufferAllocOp alloc : allocsToDelete)
      alloc.erase();
  }
};
} // namespace M::MGP
