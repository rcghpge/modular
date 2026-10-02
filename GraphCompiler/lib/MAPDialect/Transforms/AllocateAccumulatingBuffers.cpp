//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Gives every fused-output `map.iter.epilogue` a real store-target tensor via
// its `$intermediateTensor` operand, so `MAPToMOGG` never builds the
// kernel-facing `ManagedTensorSlice` over a null pointer.
//
// Normally the source is the tensor the epilogue stores into. A mixed-precision
// accumulating kernel (`containsMatmulOrConvOrConcat` whose epilogue changes
// dtype) must accumulate in the pre-cast dtype, so a fresh buffer of that dtype
// is added as an extra kernel result and used instead; the store target stays
// the real (post-cast) output.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"

#include "mlir/IR/Builders.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_ALLOCATEACCUMULATINGBUFFERS
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// Matmul/conv/concat accumulate into their output, so they must accumulate in
// their own dtype even under a dtype-changing epilogue. TODO(#22349): remove
// once those respect the fusion contract.
static bool containsMatmulOrConvOrConcat(IterOpaqueOp op) {
  for (auto callOp : op.getBody().front().getOps<CallOpaqueOp>()) {
    StringRef opName = callOp.getKernelName();
    if (opName.contains(MO::ConcatOp::getOperationName()) ||
        opName.contains(MO::MatmulOp::getOperationName()) ||
        opName.contains(MO::BatchMatmulOp::getOperationName()) ||
        opName.contains(MO::ConvOp::getOperationName()) ||
        opName.contains(MO::ConvTransposeOp::getOperationName()))
      return true;
  }
  return false;
}

// The outer tensor the epilogue's terminal `map.store` writes through. At MAP
// level the body always ends with the store.
static Value getStoreTarget(IterEpilogueOp epi) {
  auto storeOp = dyn_cast<StoreOp>(epi.getBody().front().back());
  if (!storeOp)
    return {};
  auto targetArg = dyn_cast<BlockArgument>(storeOp.getTensor());
  if (!targetArg)
    return {};
  return cast<CapturingFusionClosure>(epi.getOperation())
      .getCaptureFor(targetArg);
}

// Replaces `op` with a copy that has one extra non-chain result of
// `scratchType` (before any trailing chain) and a matching trailing output
// block argument, returned in `scratchArg`. `map.iter.opaque` cannot grow
// results in place.
static IterOpaqueOp addScratchResult(IterOpaqueOp op, Type scratchType,
                                     BlockArgument &scratchArg) {
  OpBuilder b(op);
  Value chain = op.getOutputChain();
  SmallVector<Type> resultTypes;
  for (Value r : op.getResults())
    if (r != chain)
      resultTypes.push_back(r.getType());
  resultTypes.push_back(scratchType);
  if (chain)
    resultTypes.push_back(chain.getType());

  auto newOp = IterOpaqueOp::create(b, op.getLoc(), resultTypes, op.getInputs(),
                                    op.getInputChain(),
                                    /*fusionInfo=*/Attribute());
  newOp.setDeviceRef(MO::getDeviceOrDefaultAny(op));
  if (Attribute fusionInfo = op.getFusionInfoAttr())
    newOp.setFusionInfoAttr(fusionInfo);
  newOp.setHasReadEffect(op.getHasReadEffect());
  newOp.setHasWriteEffect(op.getHasWriteEffect());
  newOp.setIsAsync(op.getIsAsync());

  newOp.getBody().takeBody(op.getBody());
  scratchArg = newOp.getBody().front().addArgument(scratchType, op.getLoc());

  unsigned newIdx = 0;
  for (Value r : op.getResults()) {
    if (r == chain)
      continue;
    r.replaceAllUsesWith(newOp.getResult(newIdx++));
  }
  if (chain)
    chain.replaceAllUsesWith(newOp.getResults().back());
  op.erase();
  return newOp;
}

struct AllocateAccumulatingBuffersPass
    : public impl::AllocateAccumulatingBuffersBase<
          AllocateAccumulatingBuffersPass> {
  using Base::Base;

  // Fails only on an unconvertible scratch dtype. The mixed-precision path
  // replaces the kernel, so it recurses into the replacement.
  LogicalResult assignIntermediateTensors(IterOpaqueOp kernel) {
    for (IterEpilogueOp epi :
         kernel.getBody().front().getOps<IterEpilogueOp>()) {
      if (epi.getIntermediateTensor())
        continue;
      Value storeTarget = getStoreTarget(epi);
      if (!storeTarget)
        continue;

      Type inElemType = epi.getBody().front().getArgument(0).getType();
      auto resultTy = dyn_cast<MO::TensorType>(epi.getResult().getType());
      bool dtypeChanges = resultTy && inElemType != resultTy.getElementType();

      if (dtypeChanges && containsMatmulOrConvOrConcat(kernel)) {
        ErrorOr<DType> inDType = MO::fromMLIRType(inElemType);
        if (inDType.isError())
          return epi.emitOpError("AllocateAccumulatingBuffers: ")
                 << inDType.getError();
        Type scratchType = resultTy.cloneWith(*inDType);
        BlockArgument scratchArg;
        IterOpaqueOp newKernel =
            addScratchResult(kernel, scratchType, scratchArg);
        epi.getIntermediateTensorMutable().assign(scratchArg);
        return assignIntermediateTensors(newKernel);
      }

      epi.getIntermediateTensorMutable().assign(storeTarget);
    }
    return success();
  }

  void runOnOperation() override {
    // Walk kernels, not epilogues: the mixed-precision path replaces the
    // kernel, and erasing the visited op is walk-safe while erasing an ancestor
    // is not. `skip()` keeps the walk out of the possibly replaced body.
    WalkResult result = getOperation().walk<WalkOrder::PreOrder>(
        [&](IterOpaqueOp kernel) -> WalkResult {
          if (failed(assignIntermediateTensors(kernel)))
            return WalkResult::interrupt();
          return WalkResult::skip();
        });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

} // namespace M::MAP
