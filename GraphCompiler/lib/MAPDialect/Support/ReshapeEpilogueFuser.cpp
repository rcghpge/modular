//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Support/ReshapeEpilogueFuser.h"

#include "GraphCompiler/MAPDialect/Dialect/MAPAttrs.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/Support/DataflowUtils.h"

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSERESHAPEEPILOGUE
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// Every `map.iter.opaque` produced today wraps each result in exactly one
// `map.iter.epilogue` callback. Finds the one for `resultIdx`, or null if
// there is none (e.g. the result is a chain, not a tensor).
static IterEpilogueOp getEpilogueForResult(IterOpaqueOp op,
                                           unsigned resultIdx) {
  Value outArg = op.getOutputBlockArgs()[resultIdx];
  for (auto epilogueOp : op.getBody().front().getOps<IterEpilogueOp>())
    if (llvm::is_contained(epilogueOp.getCaptures(), outArg))
      return epilogueOp;
  return nullptr;
}

// The single `map.index.reshape` that transforms `src`'s loop index into the
// index at which `src` loads its fused input `fusionArg`. Null unless every
// use of `fusionArg` is a `map.load` whose index is that one reshape applied
// to the loop's own (untransformed) index. Composing the reshape with another
// transform, or reading the fused input directly, is left for the plain
// `EpilogueFuser`.
static ReshapeIndexOp reshapeOnFusedInput(Block &srcBody,
                                          BlockArgument fusionArg) {
  BlockArgument loopIdx = srcBody.getArguments().back();
  ReshapeIndexOp found = nullptr;
  for (Operation *user : fusionArg.getUsers()) {
    auto load = dyn_cast<LoadOp>(user);
    if (!load)
      return nullptr;
    auto reshape = load.getIndex().getDefiningOp<ReshapeIndexOp>();
    if (!reshape || reshape.getIndex() != loopIdx)
      return nullptr;
    if (found && found != reshape)
      return nullptr;
    found = reshape;
  }
  return found;
}

// A pure reshape kernel stores exactly the value it loaded from its input,
// with no per-element compute in between. Relocating that into an epilogue is
// a no-op relabel in MGP that would only add per-element index math, so it is
// declined -- reshape fusion exists solely to unlock downstream elementwise
// ops, and there is none here.
static bool isPureReshape(Block &srcBody, BlockArgument fusionArg) {
  auto store = cast<StoreOp>(srcBody.back());
  auto load = store.getValue().getDefiningOp<LoadOp>();
  return load && load.getTensor() == fusionArg;
}

// Splices `src`'s per-element compute into `epilogueOp`, moving the reshape
// `src` applies to its load of `dst`'s result onto the epilogue's STORE index
// (with the signature's in/out shapes swapped -- the store direction is the
// inverse of the load direction). Mirrors `EpilogueFuser::fuseEpilogueOnce`,
// differing only in that relocation and the resulting shape retype.
static void fuseReshapeEpilogueOnce(IterOpaqueOp dst, unsigned resultIdx,
                                    IterEpilogueOp epilogueOp,
                                    IterForeachOp src, ReshapeIndexOp reshape,
                                    PatternRewriter &rewriter) {
  MLIRContext *ctx = dst.getContext();
  Value dstResult = dst->getResult(resultIdx);
  Block &srcBody = src.getBody().front();
  Block &epilogueBody = epilogueOp.getBody().front();

  auto epilogueStore = cast<StoreOp>(epilogueBody.back());
  Value val = epilogueStore.getValue();      // producer element for this index
  Value storeIdx = epilogueStore.getIndex(); // producer (out-shape) index
  Value cap = epilogueStore.getTensor();
  Location storeLoc = epilogueStore.getLoc();

  // Build the swapped reshape spec: the store projects the producer's
  // (out-shape) index to the consumer's (in-shape) index -- the inverse of
  // the load direction the embedded reshape encodes -- obtained by exchanging
  // the signature's in/out shapes. Sound because a reshape preserves the
  // row-major linear offset.
  MAP::ViewSignatureAttr origSig = reshape.getSpec().getSignature();
  auto swappedSpec = MAP::ReshapeSpecAttr::get(
      ctx, MAP::ViewSignatureAttr::get(ctx, origSig.getOut(), origSig.getIn()));

  // Emit the store-index transform just before the current store. Its result
  // lives in the consumer's index space (the swapped spec's IN shape ==
  // origSig.out).
  rewriter.setInsertionPoint(epilogueStore);
  Value reshapedIdx = MAP::ReshapeIndexOp::create(
      rewriter, storeLoc, MAP::IndexType::get(ctx, origSig.getOut().getRank()),
      swappedSpec, storeIdx);

  // Redirect `src`'s (reshape-indexed) loads of the fused input to the
  // producer element `val`; the embedded reshape op is then dead. Both hold by
  // construction: `val` IS the producer's element at `storeIdx`, and
  // `reshape(storeIdx)` de-linearizes back to that same element.
  unsigned numInputs = src.getInputs().size();
  auto fusionArgIt = llvm::find(src.getInputs(), dstResult);
  unsigned fusionArgIdx = fusionArgIt - src.getInputs().begin();
  BlockArgument fusionArg = srcBody.getArgument(fusionArgIdx);
  for (Operation *user : llvm::to_vector(fusionArg.getUsers())) {
    auto load = cast<LoadOp>(user);
    load.getResult().replaceAllUsesWith(val);
    load.erase();
  }
  reshape.erase();

  // Every OTHER input becomes a fresh epilogue capture, threaded through a new
  // shadowed input of `dst` (the epilogue is isolated from above).
  for (auto [i, input] : llvm::enumerate(src.getInputs())) {
    if (i == fusionArgIdx)
      continue;
    unsigned dstInputIdx = dst.getInputs().size();
    dst.insertInputsAt(dstInputIdx, input);
    Value dstInputArg = dst.getInputBlockArgs()[dstInputIdx];

    epilogueOp.getCapturesMutable().append(dstInputArg);
    BlockArgument capArg =
        epilogueBody.addArgument(input.getType(), input.getLoc());
    srcBody.getArgument(i).replaceAllUsesWith(capArg);
  }

  // Keep `src`'s final value; its store target arg is dropped with the body.
  // Rewire `src`'s loop index to the reshaped store index so every remaining
  // load (e.g. a bias) reads at the consumer index too.
  auto srcStore = cast<StoreOp>(srcBody.back());
  Value finalVal = srcStore.getValue();
  srcStore.erase();
  srcBody.getArgument(numInputs + 1).replaceAllUsesWith(reshapedIdx); // index
  srcBody.eraseArguments(0, srcBody.getNumArguments());

  // Retype only the STORE-TARGET path to the consumer's (reshaped) shape: the
  // opaque result and its DPS output block arg (the real destination the store
  // writes into) and the store-target capture. The epilogue's RESULT
  // (`epilogueOp.getResult()`) deliberately KEEPS the producer's shape -- it is
  // what the kernel binds as its fused output, and an epilogue-opt-in kernel
  // (e.g. matmul, rank-2 `c`) must keep seeing its own shape; the store lands
  // in the differently-shaped destination via the reshaped index. Emit binds
  // the kernel to an empty (null-pointer) slice of the epilogue result shape
  // (see TensorCreateEpilogueOp::emitMojo), so the kernel never touches the
  // real pointer/strides.
  Type fusedTy = src.getResult(0).getType();
  if (fusedTy != dstResult.getType()) {
    dstResult.setType(fusedTy);
    dst.getOutputBlockArgs()[resultIdx].setType(fusedTy);
    cap.setType(fusedTy);
  }

  epilogueStore.erase();
  rewriter.inlineBlockBefore(&srcBody, &epilogueBody, epilogueBody.end());
  rewriter.setInsertionPointToEnd(&epilogueBody);
  StoreOp::create(rewriter, storeLoc, cap, reshapedIdx, finalVal);

  src->replaceAllUsesWith(ValueRange{dstResult});
}

} // namespace

// Finds every currently-eligible (dst, src) edge in one read-only pass over
// the graph. Eligibility: `dst` is a `map.iter.opaque` whose result has a
// SINGLE use (no duplication, matching legacy epilogue fusion) which is a
// single-result non-chain `map.iter.foreach` consumer that reads `dst`'s
// result through a `map.index.reshape` of the loop index, and does more than
// just relabel (a pure reshape is left alone).
SmallVector<ReshapeEpilogueFuser::FusionCandidate>
ReshapeEpilogueFuser::collectCandidates() {
  SmallVector<FusionCandidate> candidates;
  DenseSet<Operation *> claimed;

  graphOp.getBody().walk([&](IterOpaqueOp dst) {
    // Only the non-chain results have an output block arg; indexing
    // `getOutputBlockArgs()[resultIdx]` for the trailing chain in
    // `getEpilogueForResult` would run off the end (mirrors `EpilogueFuser`).
    for (unsigned resultIdx = 0; resultIdx < dst.getNumNonChainResults();
         ++resultIdx) {
      Value result = dst->getResult(resultIdx);
      if (!result.hasOneUse())
        continue;

      IterEpilogueOp epilogueOp = getEpilogueForResult(dst, resultIdx);
      if (!epilogueOp)
        continue;

      Operation *user = result.getUses().begin()->getOwner();
      if (user->getBlock() != dst->getBlock())
        continue;

      auto src = dyn_cast<IterForeachOp>(user);
      if (!src || src.getNumResults() != 1 || claimed.contains(src))
        continue;
      if (isa<MO::ChainType>(src.getResult(0).getType()))
        continue;
      if (llvm::count(src.getInputs(), result) != 1)
        continue;

      Block &srcBody = src.getBody().front();
      auto fusionArgIt = llvm::find(src.getInputs(), result);
      BlockArgument fusionArg =
          srcBody.getArgument(fusionArgIt - src.getInputs().begin());

      ReshapeIndexOp reshape = reshapeOnFusedInput(srcBody, fusionArg);
      if (!reshape)
        continue;
      if (isPureReshape(srcBody, fusionArg))
        continue;

      // A dynamic shape fuses too: `IterEpilogueOp::lowerToMOGG` threads the
      // reshape's runtime from/to shapes into the epilogue functor as shape
      // captures and emit picks the dynamic `mogg.index.reshape` primitive. A
      // fully static shape uses the specialized static primitive with no
      // runtime shape.

      // Reject if another operand transitively depends on `dst`: fusing would
      // create a cycle.
      if (llvm::any_of(src->getOperands(), [&](Value otherOperand) {
            return otherOperand != result &&
                   dataflowGraphDependsOn(otherOperand, dst);
          }))
        continue;

      claimed.insert(src);
      candidates.push_back({dst, resultIdx, epilogueOp, src, reshape});
    }
  });

  return candidates;
}

ReshapeEpilogueFuser::ReshapeEpilogueFuser(MO::GraphOp graphOp)
    : graphOp(graphOp), candidates(collectCandidates()) {}

unsigned ReshapeEpilogueFuser::fuse() {
  PatternRewriter rewriter(graphOp.getContext());

  SmallVector<Operation *> toErase;
  for (FusionCandidate &candidate : candidates) {
    fuseReshapeEpilogueOnce(candidate.dst, candidate.resultIdx,
                            candidate.epilogueOp, candidate.src,
                            candidate.reshape, rewriter);
    candidate.dst.setFusionInfoAttr(
        MOGG::createFusionInfo<MOGG::FusionEpilogueAttr>(
            candidate.dst.getContext(),
            {candidate.dst.getOperation(), candidate.src.getOperation()},
            {static_cast<uint32_t>(candidate.resultIdx)}));
    candidate.dst->setLoc(mlir::FusedLoc::get(
        candidate.dst.getContext(),
        {candidate.src->getLoc(), candidate.dst->getLoc()}));
    toErase.push_back(candidate.src);
  }
  for (Operation *op : toErase)
    op->erase();

  return candidates.size();
}

namespace {

struct FuseReshapeEpiloguePass
    : public MAP::impl::FuseReshapeEpilogueBase<FuseReshapeEpiloguePass> {
  using Base::Base;

  void runOnOperation() override {
    MO::GraphOp graphOp = getOperation();
    unsigned numFused;
    do {
      numFused = ReshapeEpilogueFuser(graphOp).fuse();
    } while (numFused > 0);
  }
};

} // namespace

} // namespace M::MAP
