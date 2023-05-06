//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/MGPTransforms.h"

#include "Cache/CacheDialect/CacheAttrs.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPUtils.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "GraphRT/MEFDialect/MEFOps.h"
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

  /// Returns true if buffer is:
  ///  1. created by an mgp.cpu.buffer.{re}alloc.* operation...
  ///  2. ...which is gated on the 'execute' blocks argument chain
  static bool isAllocatedBuffer(ExecuteState &execState, Value buffer) {
    if (auto allocOp = buffer.getDefiningOp<MGPBufferAllocInterface>())
      return allocOp.getInChain() == execState.getRootChain();
    return false;
  }

  void runOnOperation() override {
    ModelOp model = getOperation();
    ExecuteState execState(model->getContext(), model.getExecuteBlock());

    model.getExecute().walk([&](MGP::CpuBufferConcatOp concatOp) {
      if (concatOp.getInBuffers().empty())
        // Ignore trivial concats.
        return;
      if (!llvm::all_of(concatOp.getInBuffers(), [&](Value buffer) {
            return isAllocatedBuffer(execState, buffer);
          }))
        // The concat input buffers are not explicitly allocated or are read
        // outside the concat.
        return;

      // Collects the raw size attributes for the split, one for every concat
      // input buffer.
      SmallVector<uint64_t> rawSizes;
      rawSizes.reserve(concatOp.getInBuffers().size());
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
        auto bufferType = dyn_cast<MGP::CpuBufferType>(inBuffer.getType());
        assert(bufferType);
        rawSizes.emplace_back(optSizeToRawSize(bufferType.getOptSize()));
        bufferTypes.emplace_back(bufferType);
        if (!bufferType.getOptSize()) {
          // Buffer does not have a statically known size.
          // We'll need to pass a size argument as an operand to the split.
          if (auto allocDynamicOp = dyn_cast<MGP::CpuBufferAllocDynamicOp>(
                  inBuffer.getDefiningOp())) {
            runtimeSizes.emplace_back(allocDynamicOp.getRuntimeSize());
          } else if (auto reallocDynamicOp =
                         dyn_cast<MGP::CpuBufferReallocDynamicOp>(
                             inBuffer.getDefiningOp())) {
            runtimeSizes.emplace_back(reallocDynamicOp.getRuntimeSize());
          } else if (auto allocSpecOp = dyn_cast<MGP::CpuBufferAllocSpecOp>(
                         inBuffer.getDefiningOp())) {
            execState.builder.setInsertionPointAfter(allocSpecOp);
            runtimeSizes.emplace_back(
                execState.builder.create<MGP::CpuTensorSpecSizeOp>(
                    concatOp->getLoc(),
                    /*spec=*/allocSpecOp.getRuntimeSpec()));
          } else if (auto reallocSpecOp = dyn_cast<MGP::CpuBufferReallocSpecOp>(
                         inBuffer.getDefiningOp())) {
            execState.builder.setInsertionPointAfter(reallocSpecOp);
            runtimeSizes.emplace_back(
                execState.builder.create<MGP::CpuTensorSpecSizeOp>(
                    concatOp->getLoc(),
                    /*spec=*/reallocSpecOp.getRuntimeSpec()));
          } else {
            assert(false && "unexpected buffer allocation op");
          }
        }
      }

      // We are now ready to create the split op.
      // For the moment, place it before the concat it will replace.
      execState.builder.setInsertionPoint(concatOp);
      auto splitOp = execState.builder.create<MGP::CpuBufferSplitBorrowOp>(
          concatOp->getLoc(),
          /*outBuffers=*/bufferTypes,
          /*inBuffer=*/concatOp.getOutBuffer(),
          /*runtimeSizes=*/runtimeSizes,
          /*outBufferRawSizes=*/rawSizes);

      // The input buffers are now provided by the split.
      for (auto [origInBuffer, newOutBuffer] :
           llvm::zip(concatOp.getInBuffers(), splitOp.getOutBuffers())) {
        origInBuffer.replaceAllUsesWith(newOutBuffer);
        origInBuffer.getDefiningOp()->erase();
      }

      // Short circuit the concat op chains.
      concatOp.getOutChain().replaceAllUsesWith(concatOp.getInChain());

      // The concat is now dead.
      concatOp->erase();

      // For readability, place the split input buffer allocation before the
      // split itself.
      moveAllBefore(splitOp.getInBuffer().getDefiningOp(), splitOp);

      // For readability, place the split before all of the users of its output
      // buffer.s
      moveAllBefore(splitOp, earliestUsingOp(splitOp.getOutBuffers()));
    });
  }
};

} // namespace
} // namespace M::MGP
