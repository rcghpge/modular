//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTensorInterface.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPUtils.h"
#include "GraphCompiler/MGPDialect/Transforms/FixupIsolatedFromAbove.h"
#include "GraphCompiler/MGPDialect/Transforms/Utils.h"
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
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// ConcatInPlacePass
//===----------------------------------------------------------------------===//

class ConcatInPlacePass
    : public impl::ConcatInPlacePassBase<ConcatInPlacePass> {
  using ConcatInPlacePassBase::ConcatInPlacePassBase;

  static SmallVector<Value>
  computeOffsets(OpBuilder &builder, Location loc,
                 ArrayRef<TypedValue<IndexType>> sliceSizes) {
    SmallVector<Value> result;
    result.reserve(sliceSizes.size());

    if (sliceSizes.empty())
      return result;

    // If all inputs are the same size, we can emit a sequence of independent
    // multiplication operations to compute the offset, rather than a chained
    // sequence of additions.
    if (llvm::all_equal(sliceSizes)) {
      Value sliceSize = sliceSizes.front();

      for (auto index : llvm::seq(sliceSizes.size())) {
        // Insert the slices just after the producing op for readability
        auto indexCst = mlir::index::ConstantOp::create(
            builder, loc, static_cast<int64_t>(index));

        Value offset =
            builder.createOrFold<mlir::index::MulOp>(loc, indexCst, sliceSize);
        result.push_back(offset);
      }

    } else {
      Value offset = mlir::index::ConstantOp::create(builder, loc, 0);
      result.push_back(offset);

      // For each concat input buffer...
      for (Value sliceSize : sliceSizes.drop_back()) {
        offset =
            builder.createOrFold<mlir::index::AddOp>(loc, offset, sliceSize);
        result.push_back(offset);
      }
    }

    return result;
  }

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

      BufferBuilder *bufferBuilder =
          builderContext.getBufferBuilder(concatBuffer.getType().getDevice());
      OpBuilder &builder = execState.builder;

      // Get sizes of each buffer.
      auto sliceSizes =
          llvm::map_to_vector(concatOp.getInBuffers(), [&](Value inBuffer) {
            auto alloc = inBuffer.getDefiningOp<MGP::BufferAllocOp>();
            ASSERT_STREAM(alloc, << "unexpected buffer allocation op "
                                 << *inBuffer.getDefiningOp());
            return alloc.getRuntimeSize();
          });

      // Compute the per-buffer offsets for each input buffer.
      auto offsets = computeOffsets(builder, loc, sliceSizes);

      SmallVector<Value> outBuffers;
      for (auto [inBuffer, sliceSize, offset] :
           llvm::zip_equal(concatOp.getInBuffers(), sliceSizes, offsets)) {
        if (Operation *op = inBuffer.getDefiningOp())
          execState.builder.setInsertionPoint(op);

        Value outSlice = bufferBuilder->bufferSlice(
            builder, loc, concatBuffer, offset, sliceSize, concatOp.getCtx());
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
