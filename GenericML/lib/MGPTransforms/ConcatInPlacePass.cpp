//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/MGPTransforms.h"

#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTensorInterface.h"
#include "GenericML/MGPDialect/MGPUtils.h"
#include "GenericML/MGPTransforms/FixupIsolatedFromAbove.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "GraphRT/MEFDialect/MEFOps.h"
#include "Support/AssertStream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "./Utils.h"

using namespace mlir;

namespace M::MGP {

// Generate support code for ConcatInPlacePass.
#define GEN_PASS_DEF_CONCATINPLACEPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// ConcatInPlacePass
//===----------------------------------------------------------------------===//

class ConcatInPlacePass
    : public impl::ConcatInPlacePassBase<ConcatInPlacePass> {
  using ConcatInPlacePassBase::ConcatInPlacePassBase;

  void runOnOperation() override {
    ModelOp modelOp = getOperation();
    BufferBuilderContext builderContext =
        TensorBufferDialectInterface::getAllBufferBuilders(modelOp);
    ExecuteState execState(modelOp->getContext(), modelOp.getExecuteBlock());

    modelOp.getExecute().walk([&](BufferConcatOp concatOp) {
      if (concatOp.getInBuffers().empty())
        // Ignore trivial concats.
        return;

      if (!llvm::all_of(concatOp.getInBuffers(), [&](Value buffer) {
            return buffer.getDefiningOp<MGPBufferAllocOpInterface>();
          }))
        // The concat input buffers are not explicitly allocated.
        return;

      auto allocOutputOp =
          concatOp.getOutBuffer().getDefiningOp<MGPBufferAllocOpInterface>();
      if (!allocOutputOp)
        // The concat output buffer is not explicitly allocated.
        return;

      DenseSet<Value> concatBuffers;
      for (Value inBuffer : concatOp.getInBuffers())
        concatBuffers.insert(inBuffer);
      if (concatBuffers.size() != concatOp.getInBuffers().size())
        // The concat input buffers are not distinct.
        return;

      // In the following we'll ignore the chains which guard the inBuffer
      // alloc ops on the assumption the dependent chains to the concat
      // indicate when the inBuffers are safe to read, and there's no additional
      // dependency captured by the alloc chains.

      // Collects the size attributes for the split, one for every concat
      // input buffer.
      SmallVector<std::optional<uint64_t>> optSizes;
      optSizes.reserve(concatOp.getInBuffers().size());
      // Collects the types of the split output buffers, one for every concat
      // input buffer.
      SmallVector<Type> bufferTypes;
      bufferTypes.reserve(concatOp.getInBuffers().size());
      // Collects the runtime size arguments for the split.
      // Will only be populated for input buffers which don't have a statically
      // known size, if any.
      SmallVector<Value> runtimeSizes;

      // For each concat input buffer...
      for (Value inBuffer : concatOp.getInBuffers()) {
        auto bufferType = dyn_cast<DeviceBufferType>(inBuffer.getType());
        GENERICML_ASSERT(bufferType);
        optSizes.emplace_back(bufferType.getOptSize());
        bufferTypes.emplace_back(bufferType);
        if (!bufferType.getOptSize()) {
          // Buffer does not have a statically known size.
          // We'll need to pass a size argument as an operand to the split.
          if (auto allocDynamicOp = dyn_cast<MGP::BufferAllocDynamicOp>(
                  inBuffer.getDefiningOp())) {
            runtimeSizes.emplace_back(allocDynamicOp.getRuntimeSize());
          } else if (auto reallocDynamicOp =
                         dyn_cast<MGP::BufferReallocDynamicOp>(
                             inBuffer.getDefiningOp())) {
            runtimeSizes.emplace_back(reallocDynamicOp.getRuntimeSize());
          } else {
            GENERICML_ASSERT(false) << "unexpected buffer allocation op "
                                    << *inBuffer.getDefiningOp();
          }
        }
      }

      BufferBuilder *bufferBuilder =
          builderContext.getBufferBuilder(concatOp.getDevice());

      // We need to ensure the output buffer will be available before any
      // writers to the split result buffers. The easiest way is to just reset
      // its input chain to be the root chain.
      //
      // TODO(#14781): We could be cleverer, for example by finding or
      // constructing a pre-dominator chain to all the chains which gate
      // writing to the split result buffers. That could be done by
      // DeferAllocsPass
      TypedValue<MEF::ChainType> rootChain =
          getRootChainForBlock(concatOp->getBlock());
      TypedValue<ContextType> context =
          bufferBuilder->getContextForOp(concatOp);
      execState.builder.setInsertionPoint(concatOp);
      Value nativeRootChain = bufferBuilder->toNativeChain(
          execState.builder, concatOp->getLoc(), context, rootChain);
      allocOutputOp.setInChain(nativeRootChain);

      // We are now ready to create the split op.
      // For the moment, place it before the concat it will replace.
      execState.builder.setInsertionPoint(concatOp);
      Value concatenatedBuffer = concatOp.getOutBuffer();
      auto outBuffers = bufferBuilder->splitBorrow(
          execState.builder, concatOp->getLoc(), context, nativeRootChain,
          concatenatedBuffer, runtimeSizes, optSizes);

      // The input buffers are now provided by the split.
      for (auto [origInBuffer, newOutBuffer] :
           llvm::zip(concatOp.getInBuffers(), outBuffers)) {
        origInBuffer.replaceAllUsesWith(newOutBuffer);
        origInBuffer.getDefiningOp()->erase();
      }

      // Short circuit the concat op chains.
      concatOp.getOutChain().replaceAllUsesWith(concatOp.getInChain());

      // The concat is now dead.
      concatOp->erase();

      // For readability, place the split input buffer allocation before the
      // split itself.
      moveAllBefore(concatenatedBuffer.getDefiningOp(),
                    outBuffers.front().getDefiningOp());

      // For readability, place the split before all the users of its output
      // buffers.
      moveAllBefore(outBuffers.front().getDefiningOp(),
                    earliestUsingOp(outBuffers));
    });
  }
};

} // namespace
} // namespace M::MGP
