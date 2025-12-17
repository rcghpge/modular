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
#include "GenericML/MGPTransforms/Utils.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "MLRT/GraphRT/MEFDialect/MEFOps.h"
#include "Support/AssertStream.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallPtrSet.h"

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
    ExecuteState execState(modelOp->getContext(), modelOp);

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

      // We are now ready to create the split op.
      // For the moment, place it before the concat it will replace.
      execState.builder.setInsertionPoint(concatOp);
      TypedValue<MGP::DeviceBufferType> concatBuffer = concatOp.getOutBuffer();

      auto loc = concatOp.getLoc();

      // Collects the runtime size arguments for the split.
      SmallVector<Value> runtimeSizes;

      BufferBuilder *bufferBuilder =
          builderContext.getBufferBuilder(concatBuffer.getType().getDevice());
      OpBuilder &builder = execState.builder;

      SmallVector<Value> outBuffers;
      Value offset =
          mlir::index::ConstantOp::create(builder, loc, 0).getResult();
      Value sliceSize = nullptr;

      // For each concat input buffer...
      for (Value inBuffer : concatOp.getInBuffers()) {
        // Insert the slices just after the producing op for readability
        if (auto op = inBuffer.getDefiningOp())
          execState.builder.setInsertionPoint(op);

        if (sliceSize)
          offset = mlir::index::AddOp::create(builder, loc, offset, sliceSize);

        // We'll need to pass a size argument as an operand to the slice.
        if (auto allocOp =
                dyn_cast<MGP::BufferAllocOp>(inBuffer.getDefiningOp())) {
          sliceSize = allocOp.getRuntimeSize();
        } else {
          ASSERT_STREAM(false, << "unexpected buffer allocation op "
                               << *inBuffer.getDefiningOp());
        }

        Value outSlice = bufferBuilder->bufferSlice(builder, loc, concatBuffer,
                                                    offset, sliceSize);

        outBuffers.push_back(outSlice);
      }

      // Move the combined alloc to the top for readability.
      concatBuffer.getDefiningOp()->moveBefore(
          outBuffers.front().getDefiningOp());

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
    });
  }
};

} // namespace
} // namespace M::MGP
