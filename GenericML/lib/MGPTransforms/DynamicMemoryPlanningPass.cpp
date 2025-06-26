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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <numeric>

#define DEBUG_TYPE "mgp-dynamic-memory-planning"

using namespace mlir;

namespace M::MGP {

// Generate support code for DynamicMemoryPlanningPass
#define GEN_PASS_DEF_DYNAMICMEMORYPLANNINGPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

struct AllocInfo {
  MGP::BufferAllocOp alloc;
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
  bool isTrivialOp = isa<KGEN::POP::POPDialect>(*producer->getDialect()) ||
                     isa<mlir::index::IndexDialect>(*producer->getDialect()) ||
                     isa<TensorExtractTensorSpecOp, CpuTensorSpecCreateOp,
                         BufferAllocOp, TensorSpecGetDimOp>(*producer);
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

    auto getDeviceContext = [&modelOp](DeviceRefAttr dev,
                                       ArrayRef<BlockArgument> devices) {
      for (auto [deviceRefAttr, deviceContext] :
           llvm::zip(modelOp.getDevices(), devices))
        if (deviceRefAttr == dev)
          return std::optional<Value>(deviceContext);
      return std::optional<Value>();
    };

    auto analysis = getAnalysis<BufferLifetimeAnalysisPass>();

    BufferBuilderContext builderContext =
        TensorBufferDialectInterface::getAllBufferBuilders(modelOp);

    InitState initState(modelOp);
    ExecuteState execState(modelOp->getContext(), modelOp);

    auto &initBuilder = initState.builder;
    auto &execBuilder = execState.builder;

    // Get dependency tree for the current block
    auto &tree = analysis.getDepTree(execState.execute);

    DenseMap<DeviceRefAttr, SmallVector<AllocInfo>> deviceToAllocs;

    DenseSet<Value> trivialVals;

    for (auto allocOp : execState.execute->getOps<MGP::BufferAllocOp>()) {
      // Ignore CPU allocations for now. Will be addressed by GEX-557
      if (allocOp.getDevice().getLabel() == kCPULabel)
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

      // Keep track of the LCM of all alignments so we can use this alignment
      // for the main allocation
      uint64_t totalAlignment = 1;
      // Sum all of constant-sized allocations, used as a rough lower bound
      uint64_t minimumConstantSize = 0;
      for (const auto &a : allocs) {
        runtimeSizes.push_back(a.runtimeSize);
        alignments.push_back(a.alignment);
        minPres.push_back(a.minPre);
        minPosts.push_back(a.minPost);
        maxPres.push_back(a.maxPre);
        maxPosts.push_back(a.maxPost);
        totalAlignment = std::lcm(totalAlignment, a.alignment);

        IntegerAttr intAttr;
        if (mlir::matchPattern(a.runtimeSize, mlir::m_Constant(&intAttr)))
          minimumConstantSize += intAttr.getInt();
      }

      // Make sure allocation size in init is non-zero
      minimumConstantSize = std::max(minimumConstantSize, uint64_t{1});

      auto loc = allocs[0].alloc->getLoc();
      execBuilder.setInsertionPointToStart(modelOp.getExecuteBlock());

      auto bufferBuilder = builderContext.getBufferBuilder(device);
      auto deviceContext =
          getDeviceContext(device, modelOp.getExecuteDevices());
      assert(deviceContext && "Could not find device context for device");

      auto initDeviceContext =
          getDeviceContext(device, modelOp.getInitDevices());
      assert(initDeviceContext && "Could not find device context for device");
      auto initLoc = initState.contextCreateOp->getLoc();

      // Create alloc in init using minimumConstantSize
      auto initSize = initBuilder.create<mlir::index::ConstantOp>(
          initLoc, minimumConstantSize);

      auto initAlloc = bufferBuilder->bufferAlloc(
          initBuilder, initLoc, initState.getContext(),
          initState.getFinalChain(), initSize, initDeviceContext.value(),
          totalAlignment);

      auto bufferType = initAlloc.getType();

      // Cache alloc in init
      auto [cached, cachedSlot] = bufferBuilder->cache(
          initBuilder, initState.contextCreateOp->getLoc(),
          initState.getContext(), initState.getFinalChain(),
          initState.contextCreateOp, initAlloc);

      // Final init chain should wait for buffer to be cached
      initState.updateFinalChain(cached);

      // Create BufferPlanOp in execute block
      auto bufferPlanOp = execBuilder.create<BufferPlanOp>(
          loc, runtimeSizes, alignments, minPres, minPosts, maxPres, maxPosts);

      // Get the cached memory pool buffer
      auto cachedMemoryPool = bufferBuilder->getCached(
          execBuilder, loc, execState.getContext(), execState.getRootChain(),
          cachedSlot, bufferType);

      // Get the size of the cached memory pool buffer.
      auto poolSize = bufferBuilder->bufferGetSize(
          execBuilder, loc, execState.getRootChain(), cachedMemoryPool);

      // Compare high watermark size returned by plan to size of buffer
      auto needToResize = execBuilder.create<index::CmpOp>(
          loc, index::IndexCmpPredicate::UGT, bufferPlanOp.getHighWatermark(),
          poolSize);

      // If high watermark is larger than existing buffer, reallocate and
      // re-cache the memory pool buffer. Otherwise, use the cached buffer.

      // Create IfOp
      auto chainType = MEF::ChainType::get(execBuilder.getContext());
      SmallVector<Type> returnTypes{chainType, bufferType};
      auto ifOp = execBuilder.create<GRT::IfOp>(
          loc, returnTypes,
          SmallVector<Value>({needToResize, execState.getRootChain(),
                              execState.getContext(), deviceContext.value(),
                              bufferPlanOp.getHighWatermark()}));

      auto ifBuilder = execBuilder;

      auto &thenBlock = ifOp.getThenRegion().emplaceBlock();
      auto &elseBlock = ifOp.getElseRegion().emplaceBlock();

      // Create 'then' block arguments
      auto thenChain = cast<TypedValue<MEF::ChainType>>(
          thenBlock.addArgument(chainType, ifOp->getLoc()));
      auto thenContext = cast<TypedValue<ContextType>>(thenBlock.addArgument(
          execState.getContext().getType(), ifOp->getLoc()));
      auto thenDeviceContext =
          thenBlock.addArgument(deviceContext->getType(), ifOp->getLoc());
      auto thenPoolSize =
          thenBlock.addArgument(poolSize.getType(), ifOp->getLoc());

      // Create 'else' block arguments
      auto elseChain = elseBlock.addArgument(chainType, ifOp->getLoc());
      auto elseContext = elseBlock.addArgument(execState.getContext().getType(),
                                               ifOp->getLoc());
      // elseDeviceContext
      elseBlock.addArgument(deviceContext->getType(), ifOp->getLoc());
      // elsePoolSize
      elseBlock.addArgument(poolSize.getType(), ifOp->getLoc());

      // Construct 'then' block body
      ifBuilder.setInsertionPointToEnd(&thenBlock);

      // If we need to resize, remove the old buffer.
      auto removeCache = bufferBuilder->removeCached(
          ifBuilder, ifOp->getLoc(), thenContext, thenChain, cachedSlot);

      thenChain = cast<TypedValue<MEF::ChainType>>(removeCache);

      // Allocate a larger buffer.
      auto thenAlloc = bufferBuilder->bufferAlloc(
          ifBuilder, loc, thenContext, thenChain, thenPoolSize,
          thenDeviceContext, totalAlignment);

      // Cache the newly allocated buffer
      auto recache = ifBuilder.create<BufferCacheOp>(
          ifOp->getLoc(), /*inChain=*/thenChain, /*theContext=*/thenContext,
          /*buffer=*/cast<TypedValue<DeviceBufferType>>(thenAlloc),
          /*bBufferSlot=*/cachedSlot);
      ifBuilder.create<MEF::OutputOp>(
          ifOp->getLoc(), SmallVector<Value>{recache.getOutChain(), thenAlloc});

      // Create the 'else' block body
      ifBuilder.setInsertionPointToEnd(&elseBlock);

      // Reuse the existing cached buffer
      auto elseGetCached = bufferBuilder->getCached(
          ifBuilder, ifOp->getLoc(), cast<TypedValue<ContextType>>(elseContext),
          cast<TypedValue<MEF::ChainType>>(elseChain), cachedSlot, bufferType);
      ifBuilder.create<MEF::OutputOp>(
          ifOp->getLoc(), SmallVector<Value>{elseChain, elseGetCached});

      // Use the output of the if as the memory pool buffer to slice from
      auto memPoolBuffer = ifOp.getResult(1);

      // Replace all allocations with slices into the memPoolBuffer.
      // Since all planned runtimeSizes are not data-dependent, this
      // shouldn't introduce any graph cycles.
      for (size_t i = 0; i < allocs.size(); ++i) {
        Value slice = execBuilder.create<BufferSliceOp>(
            loc, memPoolBuffer, bufferPlanOp.getOffsets()[i],
            allocs[i].runtimeSize);

        allocs[i].alloc.getOutBuffer().replaceAllUsesWith(slice);
      }

      for (size_t i = 0; i < allocs.size(); ++i)
        allocs[i].alloc.erase();
    }
  }
};
} // namespace M::MGP
