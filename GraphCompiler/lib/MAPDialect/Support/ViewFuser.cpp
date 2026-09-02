//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Support/ViewFuser.h"

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h"
#include "GraphCompiler/MOGGDialect/Analysis/SequenceFusionAnalysis.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include <limits>

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEVIEWS
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

bool isViewForeach(IterForeachOp op) {
  return op.getBody()
      .walk([&](IndexTransformOpInterface) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

namespace {

// Whether `op`'s body computes a load index through a `map.index.reshape`
// op. A reshape kernel must stay the terminal (most-upstream) link in any
// fused chain: unlike slice/broadcast/transpose, composing a reshape with
// another transform isn't safe to do arbitrarily yet (a future
// canonicalization pass owns that), so nothing may fuse into one.
bool isReshapeForeach(IterForeachOp op) {
  return op.getBody()
      .walk([&](ReshapeIndexOp) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

// Eligible to be a node in a fusion chain: either a single-result
// `map.iter.foreach` (a pure pointwise or view kernel, same eligibility as
// `ElementwiseFuser` -- a chain may legitimately mix elementwise and view
// links, e.g. `transpose(add(slice(x), y))`), which may act as either
// producer or consumer, or a `map.iter.opaque` kernel, which may only ever
// be a chain's sink: `viewKernelsAreCompatible` below always rejects it as a
// producer, so it can never be absorbed into some other consumer's subgraph.
static bool isEligibleFusionNode(Operation *op) {
  if (auto foreachOp = dyn_cast<IterForeachOp>(op))
    return foreachOp.getNumResults() == 1;
  return isa<IterOpaqueOp>(op);
}

// Mirrors ElementwiseFuser's own checkMutableKernelFusible: guards against
// fusing across a chain-carrying kernel (e.g. mo.mutable.store) when doing
// so could change effective ordering relative to another consumer of the
// same chain. Duplicated rather than shared -- small, and each fuser's own
// compatibility function is otherwise unrelated to the other's.
static bool checkMutableKernelFusible(Value srcInChain, Value srcOutChain,
                                      Value dstInChain, Value dstOutChain) {
  if (srcInChain && srcOutChain && dstInChain && dstOutChain) {
    if (dstInChain != srcOutChain)
      return false;
    if (!srcOutChain.hasOneUse())
      return false;
  }

  if (dstInChain && !dstInChain.hasOneUse())
    return false;

  return true;
}

// The existing `map.iter.prologue` or `map.iter.epilogue` capturing `dst`'s
// `argIdx`'th input, and that capture's own index within it -- null when
// this input isn't captured by either closure kind at all (e.g. read
// directly by `map.call.opaque`), is captured together with something else,
// or its capture is read more than once: any of those leaves no single,
// safely-replaceable load site to splice a view's body into.
struct ClosureCapture {
  CapturingFusionClosure closure = nullptr;
  unsigned captureIdx = 0;
};

static ClosureCapture findExistingCapture(IterOpaqueOp dst, unsigned argIdx) {
  BlockArgument dstArg = dst.getInputBlockArgs()[argIdx];
  if (!dstArg.hasOneUse())
    return {};
  auto closure = dyn_cast<CapturingFusionClosure>(*dstArg.getUsers().begin());
  if (!closure)
    return {};
  auto captureIt = llvm::find(closure.getCaptures(), dstArg);
  if (captureIt == closure.getCaptures().end())
    return {};
  unsigned captureIdx =
      static_cast<unsigned>(captureIt - closure.getCaptures().begin());
  BlockArgument capArg = closure.getCaptureBlockArgs()[captureIdx];
  if (!capArg.hasOneUse())
    return {};
  return {closure, captureIdx};
}

// Eligible when dst is itself a `map.iter.foreach` and at least one of
// src/dst is a view: a plain elementwise pair always shares an index space
// (same shape, no transform), so `ElementwiseFuser`'s simpler mechanism --
// which assumes the raw shared index needs no substitution -- already
// handles that edge on its own; this keeps the two fusers' jobs disjoint
// rather than competing for it.
//
// When dst is a `map.iter.opaque`, only a strict view may fuse in (a plain
// elementwise producer is `PrologueFuser`'s own job -- its simpler mechanism
// splices a producer's body in verbatim, which a transform op's
// `ParentOneOf` rejects inside a prologue or epilogue, so the two fusers'
// jobs stay disjoint here too), and only into a capture an existing
// prologue or epilogue already reads from: creating a new one is out of
// scope for this fuser.
static bool viewKernelsAreCompatible(Operation *src, Operation *dst) {
  auto srcOp = dyn_cast<IterForeachOp>(src);
  if (!srcOp)
    return false; // never treat a map.iter.opaque (or anything else) as a
                  // producer

  if (auto dstForeach = dyn_cast<IterForeachOp>(dst)) {
    if (!isViewForeach(srcOp) && !isViewForeach(dstForeach))
      return false;
    if (isReshapeForeach(dstForeach))
      return false;
    return checkMutableKernelFusible(
        srcOp.getInputChain(), srcOp.getOutputChain(),
        dstForeach.getInputChain(), dstForeach.getOutputChain());
  }

  auto dstOpaque = cast<IterOpaqueOp>(dst);
  if (!isViewForeach(srcOp))
    return false;
  // MAP_ReshapeIndexOp's ParentOneOf doesn't permit living inside a
  // map.iter.prologue the way slice/broadcast/transpose's do yet, so a
  // reshape view can't safely splice into one -- same reasoning as the
  // terminal-link restriction above, just for the prologue-dst case.
  if (isReshapeForeach(srcOp))
    return false;

  bool feedsAnyInput = false;
  for (auto [i, input] : llvm::enumerate(dstOpaque.getInputs())) {
    if (input != srcOp.getResult(0))
      continue;
    feedsAnyInput = true;
    if (!findExistingCapture(dstOpaque, i).closure)
      return false;
  }
  return feedsAnyInput;
}

} // namespace

ViewFuser::ViewFuser(MO::GraphOp graphOp)
    : graphOp(graphOp), subgraphs(collectCandidates()) {}

MOGG::FusionSubGraphs ViewFuser::collectCandidates() {
  MO::ParametricTopologyAnalysis topologyAnalysis(graphOp);

  // No cap: unlike ElementwiseFuser (which mirrors mogg-fuse-elementwise's
  // own bounded limit), a view is always cheap to fuse in, so there's no
  // reason to stop a chain from growing arbitrarily long.
  static constexpr uint64_t kMaximumFusionSetSize =
      std::numeric_limits<uint64_t>::max();
  MOGG::FusionSubGraphs subgraphs =
      MOGG::sourceDestinationSequenceFusionAnalysis(
          graphOp.getBody(), topologyAnalysis, isEligibleFusionNode,
          viewKernelsAreCompatible,
          // A view is just an index computation over its own (typically
          // small) input; re-deriving it once per consumer is always cheap,
          // unlike duplicating a real compute kernel -- so, unlike
          // ElementwiseFuser, there is no tensor-size cost model here: a
          // view can always be duplicated.
          [](Operation *) { return true; }, kMaximumFusionSetSize);

  for (auto &subgraph : subgraphs)
    topologyAnalysis.computeTopologicalSorting(subgraph);

  return subgraphs;
}

unsigned ViewFuser::fuse() {
  unsigned numFused = 0;
  for (ArrayRef<Operation *> subgraph : subgraphs) {
    fuseSubgraphOnce(subgraph);
    numFused += subgraph.size() - 1;
  }

  // Erase fused-away kernels once they have no remaining users. A DenseSet
  // guards against a double-erase when duplication placed the same op in
  // more than one subgraph.
  DenseSet<Operation *> erased;
  for (ArrayRef<Operation *> subgraph : subgraphs) {
    for (Operation *op : llvm::reverse(subgraph.drop_back())) {
      if (!erased.contains(op) && op->getUsers().empty()) {
        erased.insert(op);
        op->erase();
      }
    }
  }

  return numFused;
}

namespace {

// Splices `src`'s body directly into `dst`'s own body, at each of its load
// site(s).
void fuseViewIntoForeach(IterForeachOp dst, IterForeachOp src,
                         PatternRewriter &rewriter) {
  Block &dstBody = dst.getBody().front();

  // Every position in dst's current inputs fed by src's result. There can
  // be more than one (e.g. `mo.mul(src, src)`); each gets an independent
  // copy of src's compute spliced in, mirroring ElementwiseFuser.
  SmallVector<unsigned> fusionIndices;
  for (auto [i, input] : llvm::enumerate(dst.getInputs()))
    if (input == src.getResult(0))
      fusionIndices.push_back(i);
  assert(!fusionIndices.empty() &&
         "src must be a current input of dst inside its own subgraph");

  for (unsigned argIdx : llvm::reverse(fusionIndices)) {
    unsigned numSrcInputs = src.getInputs().size();

    // Pull src's own inputs in as new dst inputs, positioned directly
    // before the tensor argument they are about to replace.
    dst.insertInputsAt(argIdx, src.getInputs());
    ValueRange newDstArgs = dstBody.getArguments().slice(argIdx, numSrcInputs);
    BlockArgument oldTensorArg = dstBody.getArgument(argIdx + numSrcInputs);

    // Each load of the fused operand gets its own copy of src's compute,
    // wired to that specific load's own index -- not assumed to be shared
    // across every load of this operand the way ElementwiseFuser assumes
    // (correctly, there, since a pure elementwise chain always shares one
    // index space): two loads of the same view result need not use the
    // same index (e.g. dst is itself a view, or duplication placed src at
    // more than one distinct load site).
    for (Operation *user : llvm::to_vector(oldTensorArg.getUsers())) {
      auto load = cast<LoadOp>(user);
      Value loadIndex = load.getIndex();

      // Clone src's body so the original is left untouched: it may be
      // spliced in at more than one load site, fusion index, or subgraph.
      rewriter.cloneRegionBefore(src.getBody(), *dstBody.getParent(),
                                 dstBody.getParent()->begin());
      Block &srcClone = dstBody.getParent()->front();

      auto storeOp = cast<StoreOp>(srcClone.back());
      Value fusedVal = storeOp.getValue();
      storeOp.erase();
      srcClone.eraseArgument(numSrcInputs);

      // Inline right before `load` itself, not at the start of dst's body:
      // `loadIndex` may be the result of an earlier op in dst's own body
      // (e.g. dst's own index-transform chain, when dst is itself a
      // view), which only dominates uses from this point on, not from the
      // top.
      SmallVector<Value> fusedArgs(newDstArgs.begin(), newDstArgs.end());
      fusedArgs.push_back(loadIndex);
      rewriter.inlineBlockBefore(&srcClone, load, fusedArgs);

      load.getResult().replaceAllUsesWith(fusedVal);
      load.erase();
    }

    dst.eraseInputsAt(argIdx + numSrcInputs);
  }

  dst.setFusionInfoAttr(MOGG::createFusionInfo<MOGG::FusionViewAttr>(
      src->getContext(), {src, dst}));
  // Merge source locations so debugging/error-reporting tooling downstream
  // still sees both original ops.
  dst->setLoc(
      mlir::FusedLoc::get(src->getContext(), {src->getLoc(), dst->getLoc()}));
  // Mirrors ElementwiseFuser: dst now also executes src's body, so dst's
  // declared effects must cover whatever src had.
  dst.setHasReadEffect(src.getHasReadEffect() || dst.getHasReadEffect());
  dst.setHasWriteEffect(src.getHasWriteEffect() || dst.getHasWriteEffect());
}

// Splices `src`'s body into an existing `map.iter.prologue` or
// `map.iter.epilogue` inside `dst`, mirroring `PrologueFuser::fusePrologueOnce`
// -- the two differ only in where the load site to replace lives: a plain
// elementwise producer's own load site is one of dst's shadowed input block
// args directly, while a view's is one level further in, behind an existing
// closure's capture. Generic over which of the two closure kinds is involved
// via `CapturingFusionClosure`: an epilogue captures its own extra inputs
// (e.g. a bias tensor a store-path add reads from) the exact same way a
// prologue does, so fusing a view into one of those captures is the same
// splice regardless of which closure it lands in.
void fuseViewIntoClosure(IterOpaqueOp dst, IterForeachOp src,
                         PatternRewriter &rewriter) {
  SmallVector<unsigned> fusionIndices;
  for (auto [i, input] : llvm::enumerate(dst.getInputs()))
    if (input == src.getResult(0))
      fusionIndices.push_back(i);
  assert(!fusionIndices.empty() &&
         "src must be a current input of dst inside its own subgraph");

  unsigned numSrcInputs = src.getInputs().size();

  for (unsigned argIdx : llvm::reverse(fusionIndices)) {
    ClosureCapture capture = findExistingCapture(dst, argIdx);
    assert(capture.closure &&
           "viewKernelsAreCompatible should have verified this resolves");
    CapturingFusionClosure closure = capture.closure;
    // Captured before cloning below inserts a second block: `.front()` would
    // then mean the clone, not this body, for as long as both coexist -- so
    // captures are added manually against this reference instead of via
    // `addCapture()`, which always re-derives `.front()` at call time.
    Block &closureBody = closure.getBody().front();

    BlockArgument oldCap = closure.getCaptureBlockArgs()[capture.captureIdx];
    auto oldLoad = cast<LoadOp>(*oldCap.getUsers().begin());
    Value loadIndex = oldLoad.getIndex();

    // Clone src's body into the closure's own region (not dst's) so the
    // original is left untouched: it may be spliced in at more than one
    // load site.
    rewriter.cloneRegionBefore(src.getBody(), closure.getBody(),
                               closure.getBody().begin());
    Block &srcClone = closure.getBody().front();

    auto storeOp = cast<StoreOp>(srcClone.back());
    Value fusedVal = storeOp.getValue();
    storeOp.erase();
    srcClone.eraseArgument(numSrcInputs);

    // Pull src's own inputs in as new closure captures, fed by new dst
    // inputs: dst's body is isolated from above, and the closure lives
    // inside it, so each outer input must first become one of dst's own
    // shadowed input block args before the closure can capture it.
    SmallVector<Value> fusedArgs;
    for (Value input : src.getInputs()) {
      unsigned dstInputIdx = dst.getInputs().size();
      dst.insertInputsAt(dstInputIdx, input);
      Value dstInputArg = dst.getInputBlockArgs()[dstInputIdx];
      closure.getOperation()->insertOperands(
          closure.getOperation()->getNumOperands(), dstInputArg);
      fusedArgs.push_back(
          closureBody.addArgument(input.getType(), input.getLoc()));
    }
    fusedArgs.push_back(loadIndex);

    rewriter.inlineBlockBefore(&srcClone, oldLoad, fusedArgs);

    oldLoad.getResult().replaceAllUsesWith(fusedVal);
    oldLoad.erase();
    closure.eraseCaptureAt(capture.captureIdx);
    dst.eraseInputsAt(argIdx);

    if (isa<IterEpilogueOp>(closure.getOperation()))
      dst.setFusionInfoAttr(MOGG::createFusionInfo<MOGG::FusionEpilogueAttr>(
          src->getContext(), {src, dst}, {argIdx}));
    else
      dst.setFusionInfoAttr(MOGG::createFusionInfo<MOGG::FusionPrologueAttr>(
          src->getContext(), {src, dst}, {argIdx}));
    dst->setLoc(
        mlir::FusedLoc::get(src->getContext(), {src->getLoc(), dst->getLoc()}));
  }
}

} // namespace

void ViewFuser::fuseSubgraphOnce(ArrayRef<Operation *> subgraph) {
  Operation *dstOp = subgraph.back();
  PatternRewriter rewriter(dstOp->getContext());

  // Process producers nearest to `dst` first (reverse topological order): by
  // the time a farther producer is reached, the nearer one has already
  // spliced its own inputs directly into `dst`'s operand list, so every `src`
  // below is always found as one of `dst`'s CURRENT inputs.
  for (Operation *op : llvm::reverse(subgraph.drop_back())) {
    auto src = cast<IterForeachOp>(op);
    if (auto dstForeach = dyn_cast<IterForeachOp>(dstOp))
      fuseViewIntoForeach(dstForeach, src, rewriter);
    else
      fuseViewIntoClosure(cast<IterOpaqueOp>(dstOp), src, rewriter);
  }
}

namespace {

struct FuseViewsPass : public MAP::impl::FuseViewsBase<FuseViewsPass> {
  using Base::Base;

  void runOnOperation() override {
    MO::GraphOp graphOp = getOperation();
    unsigned numFused;
    do {
      numFused = ViewFuser(graphOp).fuse();
      numKernelsFused += numFused;
    } while (numFused > 0);
  }
};

} // namespace

} // namespace M::MAP
