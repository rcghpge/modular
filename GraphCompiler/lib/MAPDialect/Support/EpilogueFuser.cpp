//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Support/EpilogueFuser.h"

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

using namespace M;

namespace M::MAP {

namespace {

// Every `map.iter.opaque` produced today wraps each result in exactly one
// `map.iter.epilogue` callback (a trivial store-only callback until
// something gets fused into it). Finds the one for `resultIdx`, or null if
// there is none (e.g. the result is a chain, not a tensor).
static IterEpilogueOp getEpilogueForResult(IterOpaqueOp op,
                                           unsigned resultIdx) {
  Value outArg = op.getOutputBlockArgs()[resultIdx];
  for (auto epilogueOp : op.getBody().front().getOps<IterEpilogueOp>())
    if (llvm::is_contained(epilogueOp.getCaptures(), outArg))
      return epilogueOp;
  return nullptr;
}

// Whether `op` wraps a kernel that ACCUMULATES into its output rather than
// writing each element once. Such a kernel reads back what it wrote in the
// output's declared type, so retyping that output under it (see
// `fuseEpilogueOnce`) would have it accumulate in one type through a buffer
// typed as another. There is no scratch-buffer bridge for that case yet, so
// the fusion is declined instead.
//
// Matches on the REGISTERED op name (`map.call.opaque`'s `moOpName`), not on
// the callee symbol: the callee is the Mojo struct's own name, so keying off it
// would turn a rename of that struct into a silent false negative here -- and a
// false negative corrupts an accumulator, while a false positive only costs a
// fusion opportunity. Same contract, and same op list, as the MOGG-ABI
// `containsMatmulOrConvOrConcat` in `FuseEpilogues.cpp`.
//
// TODO(#22349): Remove checks when matmul/conv/concat kernels respect the
// fusion contract.
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

struct FusionCandidate {
  IterOpaqueOp dst;
  unsigned resultIdx;
  IterEpilogueOp epilogueOp;
  IterForeachOp src;
};

// Finds every currently-eligible (dst, src) edge in one read-only pass over
// the graph. `claimed` prevents the same `src` from being claimed by more
// than one `dst` in the same round: a single elementwise consumer can only
// be folded into one producer's epilogue at a time -- a real case, not just
// a theoretical one, e.g. `add(matmul1_result, matmul2_result)`, where both
// matmuls independently look like valid producers for the same `add`.
static SmallVector<FusionCandidate> collectCandidates(MO::GraphOp graphOp) {
  SmallVector<FusionCandidate> candidates;
  DenseSet<Operation *> claimed;

  graphOp.getBody().walk([&](IterOpaqueOp dst) {
    for (unsigned resultIdx = 0; resultIdx < dst.getNumResults(); ++resultIdx) {
      Value result = dst->getResult(resultIdx);
      if (!result.hasOneUse())
        continue;

      IterEpilogueOp epilogueOp = getEpilogueForResult(dst, resultIdx);
      if (!epilogueOp)
        continue;

      Operation *user = result.getUses().begin()->getOwner();
      if (user->getBlock() != dst->getBlock())
        continue;

      // Every `map.iter.foreach` produced today is a pure pointwise kernel (see
      // ElementwiseFuser.cpp) -- eligibility reduces to "is this a
      // single-result map.iter.foreach not already claimed this round".
      auto src = dyn_cast<IterForeachOp>(user);
      if (!src || src.getNumResults() != 1 || claimed.contains(src))
        continue;

      // A chain-returning consumer is `mo.mutable.store`: one result like any
      // pointwise foreach, but that result is a `!mo.chain` and its body has no
      // output block argument at all -- the buffer INPUT is the store target
      // (see `lowerMutableStoreToForeach`). MOGG DOES fuse this, retargeting
      // the store to that buffer instead of retyping the producer's output (the
      // `isa<MO::ChainType>(fusedResultTp)` branch in `FuseEpilogues.cpp`), and
      // declines only when the two share a buffer operand. MAP cannot do the
      // same yet -- `map.iter.opaque` models neither an out chain nor
      // `hasReadEffect`/`hasWriteEffect`, so the fused kernel would write to a
      // buffer while advertising no write effect and holding no chain -- and
      // fusing it regardless would retype `dst`'s tensor output to a chain and
      // index the src index arg at `numInputs + 1`, past the end of a body with
      // no output arg.
      //
      // TODO(GEX-3964): reach MOGG parity -- model the out chain and the memory
      // effects on `map.iter.opaque`, then retarget the epilogue's store to the
      // consumer's buffer capture instead of declining here.
      if (isa<MO::ChainType>(src.getResult(0).getType()))
        continue;

      // A type-changing src (e.g. mo.cast) is fusible: `fuseEpilogueOnce`
      // retypes `dst`'s output to the consumer's type along with it. That is
      // unsound for a kernel that accumulates into that output, though, so
      // those decline the fusion and leave the consumer as its own kernel.
      // TODO(GEX-3964): lift this with a scratch-buffer step that lets an
      // accumulating kernel keep writing its own type while the epilogue
      // converts on the way out.
      if (src.getResult(0).getType() != result.getType() &&
          containsMatmulOrConvOrConcat(dst))
        continue;

      // `fuseEpilogueOnce` redirects the loads of exactly ONE of `src`'s
      // tensor inputs to the scalar `dst` just produced. A `src` consuming
      // `dst`'s result through two inputs (`add(c, c)`) has no single such
      // input to redirect, so reject it here rather than letting the splice
      // discover it -- that keeps eligibility and the splice from disagreeing
      // and leaves the splice's own assert stating a true invariant instead
      // of guarding a reachable input shape.
      if (llvm::count(src.getInputs(), result) != 1)
        continue;

      // Those redirected loads get the scalar the epilogue holds for its OWN
      // index, so each must read `dst`'s result at exactly that index -- an
      // untransformed one. A body is free to transform the loop index
      // (`mo.slice` lowers to a `map.index.slice` feeding its load), and that
      // is fine for any OTHER operand -- `add(y, sliced)` still loads `y` at
      // the plain index -- but not for the FUSED one: `slice(y, ...)` reads `y`
      // at `i * step + start`, so handing it the epilogue's element at `i`
      // would silently drop the slice, besides inlining the transform under
      // `map.iter.epilogue`, whose parent constraint forbids it.
      //
      // The index arg is the body's last (ordered [inputs, outputs, index]).
      Block &srcBody = src.getBody().front();
      BlockArgument srcIndexArg = srcBody.getArguments().back();
      auto fusionArgIt = llvm::find(src.getInputs(), result);
      BlockArgument fusionArg =
          srcBody.getArgument(fusionArgIt - src.getInputs().begin());
      if (llvm::any_of(fusionArg.getUsers(), [&](Operation *user) {
            auto load = dyn_cast<LoadOp>(user);
            return !load || load.getIndex() != srcIndexArg;
          }))
        continue;

      // Ensure we don't fuse src if another operand is dependant on dst.
      // It would lead to an invalid IR with a cycle.
      if (llvm::any_of(src->getOperands(), [&](Value otherOperand) {
            return otherOperand != result &&
                   MOGG::valueDependsTransitivelyOn(otherOperand, dst);
          })) {
        continue;
      }

      claimed.insert(src);
      candidates.push_back({dst, resultIdx, epilogueOp, src});
    }
  });

  return candidates;
}

// Splices `src`'s per-element compute into `epilogueOp`, the callback that
// runs on every element `dst` produces before storing it.
static void fuseEpilogueOnce(IterOpaqueOp dst, unsigned resultIdx,
                             IterEpilogueOp epilogueOp, IterForeachOp src,
                             PatternRewriter &rewriter) {
  Value dstResult = dst->getResult(resultIdx);
  Block &srcBody = src.getBody().front();
  Block &epilogueBody = epilogueOp.getBody().front();
  // What the epilogue currently STORES, which is not the same as its raw
  // incoming value once something has already been fused in. `src`'s loads of
  // `dst`'s result must see everything the epilogue has accumulated so far --
  // redirecting them to block arg 0 would silently discard every earlier
  // round's compute (e.g. `add -> cast` would emit just the cast). On the
  // first round the callback is store-only, so the two coincide.
  Value val = cast<StoreOp>(epilogueBody.back()).getValue();
  Value idx = epilogueBody.getArgument(1);

  unsigned numInputs = src.getInputs().size();
  std::optional<unsigned> fusionArgIdx;
  for (auto [i, input] : llvm::enumerate(src.getInputs())) {
    if (input != dstResult)
      continue;
    assert(!fusionArgIdx &&
           "collectCandidates must reject a producer feeding more than one "
           "input of the same consumer");
    fusionArgIdx = i;
  }
  assert(fusionArgIdx && "src must consume dst's result");

  // The fusion operand's block arg is a *tensor* (map.iter.foreach's ordinary
  // input convention); `val` is already the *scalar* dst produced for this
  // element. Redirect every load of that tensor arg straight to `val` and
  // drop the now-pointless loads -- the tensor arg itself is dropped below,
  // once nothing references it anymore.
  // Every such load yields `val`'s type by construction: the loads read the
  // tensor `dst` produced, and `val` IS that element. This holds even when
  // `src` changes type -- only the store side retypes (below); what the
  // kernel hands the epilogue stays the producer's own element type.
  BlockArgument fusionArg = srcBody.getArgument(*fusionArgIdx);
  for (Operation *user : llvm::to_vector(fusionArg.getUsers())) {
    auto load = cast<LoadOp>(user);
    load.getResult().replaceAllUsesWith(val);
    load.erase();
  }

  // Every other input becomes a fresh epilogue capture. `dst`'s body is
  // isolated from above, and `epilogueOp` is nested inside it, so it cannot
  // capture `input` directly (an outer value) -- `input` is first threaded in
  // as a new shadowed input of `dst` itself, and that new input block arg is
  // what the epilogue actually captures.
  for (auto [i, input] : llvm::enumerate(src.getInputs())) {
    if (i == *fusionArgIdx)
      continue;
    unsigned dstInputIdx = dst.getInputs().size();
    dst.insertInputsAt(dstInputIdx, input);
    Value dstInputArg = dst.getInputBlockArgs()[dstInputIdx];

    unsigned capIdx = epilogueOp.getNumOperands();
    epilogueOp->insertOperands(capIdx, dstInputArg);
    BlockArgument cap =
        epilogueBody.addArgument(input.getType(), input.getLoc());
    srcBody.getArgument(i).replaceAllUsesWith(cap);
  }

  // The output arg's only use, the store, is erased below; the index arg
  // becomes the epilogue's own.
  auto srcStore = cast<StoreOp>(srcBody.back());
  Value finalVal = srcStore.getValue();
  srcStore.erase();
  srcBody.getArgument(numInputs + 1).replaceAllUsesWith(idx); // index arg

  // Every block argument now has zero uses; drop them all so `srcBody`'s
  // signature matches an empty substitution list below.
  srcBody.eraseArguments(0, srcBody.getNumArguments());

  // Rebuild the epilogue's trailing store against src's final computed
  // value instead of the raw `val`: erase the old one, splice in src's
  // remaining ops (loads of captures + elementwise calls) at the now-empty
  // end of the epilogue body, then re-emit the store there.
  auto epilogueStore = cast<StoreOp>(epilogueBody.back());
  Value cap = epilogueStore.getTensor();
  Value capIdx = epilogueStore.getIndex();
  Location storeLoc = epilogueStore.getLoc();
  epilogueStore.erase();

  // When `src`'s type differs from what `dst` declared, retype the whole store
  // path: the op result, its shadowed output block arg, the epilogue's result,
  // and the body arg the trailing store writes through. The epilogue's LEADING
  // value arg keeps the producer's type -- the kernel hands over what it
  // computed.
  Type fusedTy = src.getResult(0).getType();
  if (fusedTy != dstResult.getType()) {
    dstResult.setType(fusedTy);
    dst.getOutputBlockArgs()[resultIdx].setType(fusedTy);
    epilogueOp.getResult().setType(fusedTy);
    cap.setType(fusedTy);
  }

  rewriter.inlineBlockBefore(&srcBody, &epilogueBody, epilogueBody.end());

  rewriter.setInsertionPointToEnd(&epilogueBody);
  StoreOp::create(rewriter, storeLoc, cap, capIdx, finalVal);

  // src is now fully absorbed into dst's epilogue: everything that used to
  // consume src's own result should consume dst's result instead.
  src->replaceAllUsesWith(ValueRange{dstResult});
}

} // namespace

void EpilogueFuser::fuse() {
  PatternRewriter rewriter(graphOp.getContext());

  // Loop to a local fixed point so a chain of epilogue-eligible consumers
  // (matmul -> add -> relu) fully collapses, not just its first link: each
  // round only sees the sole-use edge immediately adjacent to a producer, so
  // an N-link chain needs N rounds.
  SmallVector<FusionCandidate> candidates = collectCandidates(graphOp);
  while (!candidates.empty()) {
    SmallVector<Operation *> toErase;
    for (FusionCandidate &candidate : candidates) {
      fuseEpilogueOnce(candidate.dst, candidate.resultIdx, candidate.epilogueOp,
                       candidate.src, rewriter);
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

    candidates = collectCandidates(graphOp);
  }
}

} // namespace M::MAP
