//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Gives every fused-output `map.iter.epilogue` a real store-target tensor via
// its `$intermediateTensor` operand, so `MAPToMOGG` can always build the
// kernel-facing `ManagedTensorSlice` over real storage instead of a dangling
// null pointer.
//
// Normally the source is the tensor the epilogue stores into. For a
// mixed-precision accumulating kernel (`containsMatmulOrConvOrConcat` whose
// epilogue changes dtype) the kernel must accumulate in the pre-cast dtype, so
// a fresh accumulation buffer of that dtype is added as an extra kernel result
// and used as the source instead; the store target stays the real
// (post-cast) output. Mirrors the legacy MOGG scratch mechanism in
// `FuseEpilogues.cpp` (`ScratchPlaceholdersOp` +
// `containsMatmulOrConvOrConcat`). Lifts TODO(GEX-3964).
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

// Whether the kernel wraps a matmul/conv/concat -- an op that accumulates into
// its output and so must accumulate in its own dtype even under a
// dtype-changing epilogue. Mirrors `containsMatmulOrConvOrConcat` in
// EpilogueFuser.cpp / FuseEpilogues.cpp. TODO(#22349): remove once those
// respect the fusion contract.
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

// The tensor an epilogue stores into: the outer value its terminal `map.store`
// writes through. At MAP level the body always ends with the store (the
// compute-vs-store classification happens later, in MAPToMOGG).
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

// Rebuilds `op` with one extra non-chain result of `scratchType` (kept before
// any trailing chain result) and a matching trailing output block argument.
// `map.iter.opaque` has no in-place way to grow its results, so this replaces
// the op, moving its body over. Returns the new op; `scratchArg` is the new
// output block argument (the scratch buffer, in the kernel body's scope).
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

  // Move the body over, then append the scratch output block arg: it becomes
  // the last output (the chain has no block arg), matching the new last
  // non-chain result.
  newOp.getBody().takeBody(op.getBody());
  scratchArg = newOp.getBody().front().addArgument(scratchType, op.getLoc());

  // Old non-chain results map one-to-one to the new ones; the chain moves to
  // the end. The scratch result has no prior user.
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

  // Gives each fused-output epilogue in `kernel` an `$intermediateTensor`.
  // Fails only on an unconvertible scratch dtype. The mixed-precision path
  // replaces the kernel (a `map.iter.opaque` cannot grow a result in place),
  // moving its remaining epilogues into the replacement, so on that path we
  // recurse into the new kernel and stop touching the old one.
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
        // Mixed-precision accumulating kernel: accumulate in the pre-cast dtype
        // in a fresh scratch result; the lambda still stores the cast value to
        // the real output.
        ErrorOr<DType> inDType = MO::fromMLIRType(inElemType);
        if (inDType.isError())
          return epi.emitOpError("AllocateAccumulatingBuffers: ")
                 << inDType.getError();
        Type scratchType = resultTy.cloneWith(*inDType);
        BlockArgument scratchArg;
        IterOpaqueOp newKernel =
            addScratchResult(kernel, scratchType, scratchArg);
        epi.getIntermediateTensorMutable().assign(scratchArg);
        // Growing the kernel's results replaced it: the remaining epilogues now
        // live in `newKernel`, so continue there rather than on the erased op.
        return assignIntermediateTensors(newKernel);
      }

      // Ordinary case: the store target already provides a valid source.
      epi.getIntermediateTensorMutable().assign(storeTarget);
    }
    return success();
  }

  void runOnOperation() override {
    // Walk kernels, not epilogues: the mixed-precision path replaces the parent
    // `map.iter.opaque`, and erasing the op currently being visited is
    // walk-safe (the traversal pre-fetches the next sibling) while erasing an
    // ancestor mid-walk is not. Pre-order + `skip()` keeps us out of the (now
    // possibly replaced) kernel body, whose epilogues
    // `assignIntermediateTensors` handled.
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
