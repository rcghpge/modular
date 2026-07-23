//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEVIEWS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

/**
1. Fuse compatible view ops into other kernels.
2. Lower `mo.static.reshape` ops into a kernel and fuse them into other kernels.
The reason this is done as a separate step is because reshape ops are not
lowered and they must be the last view op fused into a kernel because the kernel
implementation expects contiguous strides. Therefore, bcast -> reshape will
yield invalid results.
3. Raise mogg.kernels that used to be mo.static.reshape back into reshapes. This
step is optional for correctness but necessary for performance. Reshapes, as
with other views, copy data. A later pass, MOGGToMGP, will turn the reshape
into a primitive that does not copy.
4. Raise unfused contiguous mo.slice kernels back to mogg.slice so that
MOGGToMGP can lower them as zero-copy buffer slices.

All in all, this pass fuses views into other views or kernels. Then, reshapes
are fused and the remaining ones will be later lowered to a fast primitive.
Unfused contiguous slices are raised to mogg.slice for zero-copy lowering.

GUARANTEES:
- All remaining reshapes are `mo.static.reshape.
- No view fuses into a reshape kernel.
 */

template <typename OpTy>
bool isViewKernelOfType(Operation *op) {
  auto kernel = dyn_cast<MOGG::KernelOp>(op);
  if (!kernel || !kernel.isViewOp())
    return false;

  SmallVector<CallViewOp> callViews =
      llvm::to_vector(kernel.getExecBody().getOps<CallViewOp>());

  // This is a unfused reshape operations.
  if (callViews.size() != 1)
    return false;

  if (callViews.front().getMoOpName() != OpTy::getOperationName())
    return false;

  return true;
}

/// Returns the first non-size-1 dimension index of inputType, or rank if all
/// dimensions have size 1.
static uint64_t findSliceDim(MO::TensorType inputType) {
  auto inputShape = inputType.getTypedShapeAttr().getValues();
  uint64_t k = inputType.getRank();
  for (uint64_t i = 0; i < inputType.getRank(); ++i) {
    auto asInt = dyn_cast<IntegerAttr>(inputShape[i]);
    if (!asInt || asInt.getValue().getSExtValue() != 1) {
      k = i;
      break;
    }
  }
  return k;
}

/// Returns true if op is a view kernel wrapping a contiguous mo.slice (step=1
/// along the sliced dimension, all other dims unsliced).  Used in
/// canFuseOperand to force-fuse contiguous slices into their consumers even
/// when the consumer does not opt into general input fusion.
static bool isContiguousSliceKernel(Operation *op) {
  if (!isViewKernelOfType<MO::SliceOp>(op))
    return false;

  auto kernel = cast<KernelOp>(op);
  auto callViewOp = *kernel.getExecBody().getOps<CallViewOp>().begin();

  auto inputValue = callViewOp->getOperand(0);
  auto inputType = cast<MO::TensorType>(inputValue.getType());
  auto outputType = cast<MO::TensorType>(callViewOp->getResult(0).getType());
  auto stepValue = kernel.getOperand(
      cast<BlockArgument>(callViewOp->getOperand(3)).getArgNumber());

  auto constStep = stepValue.getDefiningOp<MO::ConstantOp>();
  if (!constStep)
    return false;

  auto maybeStepValue = MO::tryGetConstantBlob<int64_t>(
      constStep.getValue(), constStep.getType().getDtype());
  if (!maybeStepValue)
    return false;

  auto inputShape = inputType.getTypedShapeAttr().getValues();
  auto outputShape = outputType.getTypedShapeAttr().getValues();
  uint64_t k = findSliceDim(inputType);

  if (k < inputType.getRank() && (*maybeStepValue)[k] != 1)
    return false;

  for (uint64_t i = 0; i < inputType.getRank(); ++i) {
    if (i == k)
      continue;
    if (inputShape[i] != outputShape[i])
      return false;
  }

  return true;
}

void raiseBackReshapeKernels(MLIRContext *ctx,
                             ArrayRef<KernelOp> reshapeKernels) {
  OpBuilder builder(ctx);
  for (auto k : reshapeKernels) {
    builder.setInsertionPointAfter(k);
    auto reshapeOp = MO::Static::ReshapeOp::create(
        builder, k->getLoc(), k->getResultTypes(), k.getOperands());

    k->replaceAllUsesWith(reshapeOp);
    k->erase();
  }
}

/// Raise unfused contiguous slice kernels back to mogg.slice ops.
/// mogg.slice is a zero-copy buffer slice that MOGGToMGP will lower to a
/// mgp.tensor.slice primitive. Only slices with constant start/stop[k] can be
/// raised; dynamic-start slices remain as copying kernels.
void raiseContiguousSliceKernels(MLIRContext *ctx, MO::GraphOp graph) {
  SmallVector<KernelOp> sliceKernels;
  graph.walk([&](KernelOp op) {
    if (isContiguousSliceKernel(op))
      sliceKernels.push_back(op);
  });

  OpBuilder builder(ctx);
  for (auto kernelOp : sliceKernels) {
    // Skip if all uses were fused (result is dead).
    if (kernelOp->getResults().front().use_empty())
      continue;

    auto callViewOp = *kernelOp.getExecBody().getOps<CallViewOp>().begin();

    // Map exec-body block arguments back to kernel's external operands.
    auto getExternalOperand = [&](Value v) -> Value {
      if (auto blockArg = dyn_cast<BlockArgument>(v))
        return kernelOp.getOperand(blockArg.getArgNumber());
      return v;
    };

    Value input = getExternalOperand(callViewOp->getOperand(0));
    Value startVec = getExternalOperand(callViewOp->getOperand(1));
    Value stopVec = getExternalOperand(callViewOp->getOperand(2));

    auto inputType = cast<MO::TensorType>(input.getType());
    uint64_t k = findSliceDim(inputType);

    // Only raise when start[k] and stop[k] are compile-time constants.
    // Dynamic-start slices remain as copying kernels.
    auto maybeStart =
        MO::tryGetConstantIntVector(cast<MO::TensorValue>(startVec));
    auto maybeStop =
        MO::tryGetConstantIntVector(cast<MO::TensorValue>(stopVec));
    if (!maybeStart || !maybeStop)
      continue;

    DeviceRefAttr hostDevice =
        cast<MO::TensorType>(startVec.getType()).getDeviceRef();

    builder.setInsertionPointAfter(kernelOp);
    Value startK = MO::ConstantOp::create(builder, kernelOp->getLoc(),
                                          (*maybeStart)[k], hostDevice);
    Value stopK = MO::ConstantOp::create(builder, kernelOp->getLoc(),
                                         (*maybeStop)[k], hostDevice);

    auto resultType = cast<MO::TensorType>(kernelOp->getResultTypes()[0]);
    auto sliceOp = MOGG::SliceOp::create(builder, kernelOp->getLoc(),
                                         resultType, input, startK, stopK);

    kernelOp->replaceAllUsesWith(sliceOp->getResults());
    kernelOp->erase();
  }
}

struct FuseViewsPass : public impl::FuseViewsBase<FuseViewsPass> {
  using FuseViewsBase::FuseViewsBase;

  void runOnOperationImpl(MO::GraphOp graph, MLIRContext *ctx,
                          bool fuseReshape) {
    graph.walk([&](MOGG::KernelOp op) {
      // We never fuse other kernel into reshapes since reshape requires a
      // contiguous buffer.
      if (isViewKernelOfType<MO::Static::ReshapeOp>(op))
        return;

      // Traverse from the back to simplify removing operands.
      int32_t index = op->getNumOperands() - 1;
      while (index >= 0) {
        if (!canFuseOperand(index, op)) {
          index--;
          continue;
        }

        bool isReshape = isViewKernelOfType<MO::Static::ReshapeOp>(
            (op->getOperand(index).getDefiningOp()));
        // if we are not fusing in reshapes, skip fusible reshape operand.
        // If we are fusing in reshapes, then we skip things that aren't
        // reshapes. This avoids cases like:
        //   non-reshape -> reshape -> non-reshape
        //   non-reshape -> reshape+non-reshape (first iteration)
        //   non-reshape+reshape+non-reshape  (second iteration)
        if (fuseReshape != isReshape) {
          index--;
          continue;
        }

        fuseInputAtIndex(op, index);
        index = op->getNumOperands() - 1;
      }
    });
  }

  /// Return the index of the input to fuse or -1 if nothing can be fused.
  bool canFuseOperand(int32_t index, Operation *op) {
    // Block arguments have no defining op and obviously cant be fused.
    Operation *edge = op->getOperand(index).getDefiningOp();
    if (!edge)
      return false;

    // Only fuse within the same block and not across, i.e, control flow.
    if (edge->getBlock() != op->getBlock())
      return false;

    auto dstKernel = dyn_cast<MOGG::KernelOp>(op);
    if (!dstKernel)
      return false;

    if (auto alloc = dstKernel.getAllocIndices(); alloc && !alloc.empty())
      return false;

    auto srcKernel = dyn_cast<MOGG::KernelOp>(edge);
    if (!srcKernel)
      return false;

    // Only fuse view operations (when will a view kernel returns more than
    // one result, probably dead code, or should we instead verify it during
    // IR construction?)
    if (!srcKernel.isViewOp() || srcKernel.getNumResults() != 1)
      return false;

    // If the kernel in which we fuse is a view, always fuse.
    if (dstKernel.isViewOp())
      return true;

    // If the kernel in which we fuse opted into input fusion, always fuse.
    if (dstKernel.isOperandFusible(index, true))
      return true;

    // If the kernel did not opt into input fusion but the view being fused is a
    // contiguous slice, always fuse.  Unfused contiguous slice kernels are
    // raised back to mogg.slice (zero-copy) by raiseContiguousSliceKernels.
    if (isContiguousSliceKernel(srcKernel))
      return true;

    // The kernel opted out of fusion and the view being fused would have
    // non-contiguous strides. Do not fuse.
    return false;
  }

  void fuseInputAtIndex(MOGG::KernelOp kernel, int32_t idx) {
    ++numberOfFusedViews;

    Value toFuse = kernel.getOperand(idx);
    auto viewProducer = cast<MOGG::KernelOp>(toFuse.getDefiningOp());
    kernel.insertInputsAt(idx + 1, viewProducer.getInputs());

    IRRewriter rewriter(kernel->getContext());
    rewriter.cloneRegionBefore(viewProducer.getExecBody(),
                               &kernel.getExecBody().front());

    Block *src = &kernel.getExecBody().front();
    Block *dst = &kernel.getExecBody().back();
    auto srcTerm = cast<MOGG::CallMaterializeOp>(src->getTerminator());

    ValueRange inlinedArgs = dst->getArguments().slice(
        idx + 1, viewProducer.getExecBody().getNumArguments());

    rewriter.inlineBlockBefore(src, dst, dst->begin(), inlinedArgs);

    // Clean up.
    dst->getArgument(idx).replaceAllUsesWith(srcTerm.getInput());
    kernel.eraseInputsAt(idx);
    auto srcOutputPlaceholders = srcTerm.getOutput().getDefiningOp();
    srcTerm.erase();
    srcOutputPlaceholders->erase();

    kernel.setFusionInfoAttr(createFusionInfo<FusionViewAttr>(
        kernel->getContext(), {viewProducer, kernel}));
    // Merge source locations so that Python stack traces from fused-in ops
    // survive into AttachPythonStackTraces.
    kernel->setLoc(mlir::FusedLoc::get(
        kernel->getContext(), {viewProducer->getLoc(), kernel->getLoc()}));
  }

  void runOnOperation() override {
    auto graph = getOperation();
    auto ctx = graph.getContext();

    // Fuse all view kernels (excluding reshapes).
    runOnOperationImpl(graph, ctx, /*fuseReshape=*/false);

    // Then fuse reshapes.
    SmallVector<KernelOp> reshapes;
    graph.walk([&](MOGG::KernelOp op) {
      if (isViewKernelOfType<MO::Static::ReshapeOp>(op))
        reshapes.push_back(op);
    });

    if (!reshapes.empty())
      runOnOperationImpl(graph, ctx, /*fuseReshape=*/true);

    // Raise unfused reshapes back to a MO op to be eventually lowered
    // as MGP primitives (a no-op, really).
    raiseBackReshapeKernels(ctx, reshapes);

    // Raise unfused contiguous slice kernels back to mogg.slice so that
    // MOGGToMGP can lower them as zero-copy buffer slices.
    raiseContiguousSliceKernels(ctx, graph);
  }
};

} // namespace

} // namespace M::MOGG
