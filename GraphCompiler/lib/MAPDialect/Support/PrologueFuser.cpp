//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Support/PrologueFuser.h"

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace M;

namespace M::MAP {

namespace {

struct FusionCandidate {
  IterOpaqueOp dst;
  IterPrologueOp prologueOp;
  unsigned captureIdx;
  IterForeachOp src;
  unsigned dstInputIdx;
};

// Finds every currently-eligible (prologueOp, captureIdx, src) edge in one
// read-only pass over the graph. `hasOneUse()` on src's result is the only
// gate needed against a producer being claimed twice: a value with exactly
// one use cannot simultaneously be a capture of two different prologues (or
// two different capture positions of the same one).
static SmallVector<FusionCandidate> collectCandidates(MO::GraphOp graphOp) {
  SmallVector<FusionCandidate> candidates;

  graphOp.getBody().walk([&](IterPrologueOp prologueOp) {
    auto dst = prologueOp->getParentOfType<IterOpaqueOp>();
    if (!dst)
      return;

    Block::BlockArgListType dstInputArgs = dst.getInputBlockArgs();
    for (auto [i, cap] : llvm::enumerate(prologueOp.getCaptures())) {
      // `dst`'s body is isolated from above: the prologue's own capture is
      // one of dst's shadowed input block args, not a direct reference to
      // the producer's outer SSA value -- map back through dst's own input
      // list to find the actual producer.
      auto capArg = dyn_cast<BlockArgument>(cap);
      if (!capArg)
        continue;
      auto argIt = llvm::find(dstInputArgs, capArg);
      if (argIt == dstInputArgs.end())
        continue;
      unsigned dstInputIdx =
          static_cast<unsigned>(argIt - dstInputArgs.begin());

      Value producerVal = dst.getInputs()[dstInputIdx];
      Operation *producer = producerVal.getDefiningOp();
      if (!producer)
        continue;

      if (producer->getBlock() != dst->getBlock())
        continue;

      // `producerVal` must feed dst through exactly this one shadowed input
      // (not also feed some other input, or anything outside dst), and that
      // shadowed input's block arg must be read by exactly this one
      // prologue capture (not aliased into any other prologue or read
      // directly by map.call.opaque) -- otherwise it is still genuinely
      // needed elsewhere, so fusion must not fire.
      if (!producerVal.hasOneUse() || !capArg.hasOneUse())
        continue;

      // Every `map.iter.foreach` produced today is a pure pointwise kernel
      // (see ElementwiseFuser.cpp) -- eligibility reduces to "is this a
      // single-result map.iter.foreach".
      auto src = dyn_cast<IterForeachOp>(producer);
      if (!src || src.getNumResults() != 1)
        continue;

      // Same-device guard: compares the device the captured value's own type
      // carries against the device `src` actually executes on. Both go through
      // `getDeviceOrDefault` so an unannotated (implicit host) device on one
      // side still compares equal to an explicit host device ref on the other.
      // `IterOpaqueOp` carries no device of its own, so the operand's type is
      // what stands in for the consumer here.
      auto capturedTensorTy = dyn_cast<MO::TensorType>(cap.getType());
      if (!capturedTensorTy || MO::getDeviceOrDefault(capturedTensorTy) !=
                                   MO::getDeviceOrDefault(src))
        continue;

      candidates.push_back(
          {dst, prologueOp, static_cast<unsigned>(i), src, dstInputIdx});
    }
  });

  return candidates;
}

// Splices `src`'s per-element compute into `prologueOp`, the callback that
// runs to produce the value `dst` reads for this element, in place of a
// separately materialized tensor.
static void fusePrologueOnce(IterOpaqueOp dst, IterPrologueOp prologueOp,
                             unsigned captureIdx, IterForeachOp src,
                             PatternRewriter &rewriter) {
  Block &srcBody = src.getBody().front();
  Block &prologueBody = prologueOp.getBody().front();
  Value idx = prologueBody.getArgument(0);
  unsigned numSrcInputs = src.getInputs().size();

  // The capture being fused away, and the trivial load standing in for its
  // (not yet fused) materialized value.
  BlockArgument oldCap = prologueBody.getArgument(1 + captureIdx);
  assert(oldCap.hasOneUse() &&
         "a prologue capture should be read exactly once before fusion");
  auto oldLoad = cast<LoadOp>(*oldCap.getUsers().begin());

  // The splice below replaces that load's result with the value `src` stores,
  // so the two must agree on type -- and they do by construction, with no
  // retyping needed on this side: a dtype-changing producer (`mo.cast`) casts
  // BEFORE the kernel reads, so `src`'s result type, hence the operand type
  // `dst` declares, hence this load, is already the post-cast type. (The
  // epilogue side has to retype its store path precisely because there the
  // conversion happens after the kernel writes.)
  Value fusedVal = cast<StoreOp>(srcBody.back()).getValue();
  if (oldLoad.getResult().getType() != fusedVal.getType())
    llvm_unreachable("prologue fusion across mismatched element types");

  // Pull src's own inputs in as new prologue captures, loaded explicitly --
  // the isolated prologue body has no ambient access to them. `dst`'s own
  // body is isolated from above too, and prologueOp lives inside it, so each
  // outer input must first become one of dst's own shadowed input block args
  // before prologueOp can capture it.
  SmallVector<Value> fusedArgs;
  for (Value input : src.getInputs()) {
    unsigned dstInputIdx = dst.getInputs().size();
    dst.insertInputsAt(dstInputIdx, input);
    Value dstInputArg = dst.getInputBlockArgs()[dstInputIdx];

    unsigned newCapIdx = prologueOp.getNumOperands();
    prologueOp->insertOperands(newCapIdx, dstInputArg);
    fusedArgs.push_back(
        prologueBody.addArgument(input.getType(), input.getLoc()));
  }
  fusedArgs.push_back(idx); // shared index

  // src's own output tensor arg is dead once its store is erased.
  srcBody.back().erase();
  srcBody.eraseArgument(numSrcInputs);

  // The trivial load's result is exactly what src would have computed;
  // redirect whatever consumed it (the prologue's yield) to src's
  // freshly-inlined final value, then drop the now-pointless load and the
  // now-fully-unused capture it read from.
  oldLoad.getResult().replaceAllUsesWith(fusedVal);
  oldLoad.erase();
  prologueOp.getCapturesMutable().erase(captureIdx);
  prologueBody.eraseArgument(1 + captureIdx);

  // Inline src's remaining ops (loads of its own inputs + elementwise call)
  // at the front of the prologue body, wiring its own inputs to the new
  // captures above and reusing the prologue's shared index.
  rewriter.inlineBlockBefore(&srcBody, &prologueBody, prologueBody.begin(),
                             fusedArgs);

  // dst no longer depends on src's materialized result at all once this
  // fusion absorbs it: dst's body is isolated from above, so `src`'s result
  // only ever reached it through the shadowed input block arg above, which
  // is now unused (its one use, this prologue's capture, was just erased).
  // eraseInputsAt drops both the operand and that now-dead block arg
  // together, keeping the two in sync.
  for (auto [i, input] : llvm::enumerate(dst.getInputs())) {
    if (input == src.getResult(0)) {
      dst.eraseInputsAt(i);
      break;
    }
  }
}

} // namespace

void PrologueFuser::fuse() {
  PatternRewriter rewriter(graphOp.getContext());

  // One pass is enough, unlike `EpilogueFuser`'s loop to a fixed point: fusing
  // a producer in only ever gives `dst` plain new inputs, never a new
  // prologue-fusible one, so no fusion can expose a candidate a second pass
  // would find. Collapsing a longer elementwise chain into the single producer
  // this sees is `ElementwiseFuser`'s job, which is why it must run first.
  SmallVector<FusionCandidate> candidates = collectCandidates(graphOp);

  // Highest input index first: `createFusionInfo` PREPENDS each new (leaf,
  // operand index) pair to the group it merges into, so descending order is
  // what leaves the recorded `operandIndices` ascending and paired with their
  // leaves.
  llvm::stable_sort(candidates,
                    [](const FusionCandidate &a, const FusionCandidate &b) {
                      return a.dstInputIdx > b.dstInputIdx;
                    });

  SmallVector<Operation *> toErase;
  for (FusionCandidate &candidate : candidates) {
    fusePrologueOnce(candidate.dst, candidate.prologueOp, candidate.captureIdx,
                     candidate.src, rewriter);
    candidate.dst.setFusionInfoAttr(
        MOGG::createFusionInfo<MOGG::FusionPrologueAttr>(
            candidate.dst.getContext(),
            {candidate.src.getOperation(), candidate.dst.getOperation()},
            {candidate.dstInputIdx}));
    candidate.dst->setLoc(mlir::FusedLoc::get(
        candidate.dst.getContext(),
        {candidate.src->getLoc(), candidate.dst->getLoc()}));
    toErase.push_back(candidate.src);
  }
  for (Operation *op : toErase)
    op->erase();
}

} // namespace M::MAP
