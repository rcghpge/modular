//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Hoists chains of index-transform ops (ops implementing
// MAP::IndexTransformOpInterface, e.g. map.index.slice) out of any
// CapturingFusionClosure (map.iter.foreach.body, map.iter.prologue,
// map.iter.epilogue) into a chain of "stride" ops (e.g.
// map.index.slice.stride) in the enclosing map.iter.opaque, so the transform
// runs once per kernel invocation instead of once per loop index.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/Support/DataflowUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_LOOPINVARIANTVIEWMOTION
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// Walks backward from `index` through a chain of index-transform ops,
// closest-to-load first, stopping at the first value that isn't one
// (typically the closure's own index block argument).
SmallVector<IndexTransformOpInterface> collectIndexTransformChain(Value index) {
  SmallVector<IndexTransformOpInterface> chain;
  Value current = index;
  while (Operation *def = current.getDefiningOp()) {
    auto transform = dyn_cast<IndexTransformOpInterface>(def);
    if (!transform)
      break;
    chain.push_back(transform);
    current = transform.getIndex();
  }
  return chain;
}

// Whether `op`'s result is used -- directly, or transitively through further
// index-transform ops -- as a `map.store` index. Such an op projects the loop
// index to a store's target position: it maps an INDEX, not a tensor, so it
// cannot be hoisted into a materialized `.stride` view and must stay inside the
// closure, along with any op it feeds that store index through. A load-side
// view that only ever reaches loads does NOT feed a store and is hoistable --
// even when it is composed on top of a store-index transform (e.g. a broadcast
// fused onto a reshaped store index: the broadcast feeds only its load, so it
// hoists, while the reshape feeds the store and stays).
//
// The consumer walk stays in index space: it recurses through index-transform
// users (advance) but prunes anything else (skip), so it never follows a
// `map.load`'s data result into the store that consumes it as a stored VALUE --
// only a store that consumes the index directly counts.
bool transitivelyFeedsStore(IndexTransformOpInterface op) {
  return walkDataflowGraph<WalkDirection::Consumers>(
             op.getOperation(),
             [](Operation *user) -> WalkResult {
               if (isa<StoreOp>(user))
                 return WalkResult::interrupt();
               if (isa<IndexTransformOpInterface>(user))
                 return WalkResult::advance();
               return WalkResult::skip();
             })
      .wasInterrupted();
}

// The load-side prefix of `chain` (closest-to-consumer first) that is
// hoistable: the leading run of ops that do not transitively feed a store.
// Feeding a store is monotonic down the chain -- if an op feeds a store, every
// op below it (nearer the loop index) feeds it through that op too -- so the
// hoistable ops form a contiguous prefix, and the first store-feeding op plus
// everything under it stays in the closure.
size_t hoistablePrefixLen(ArrayRef<IndexTransformOpInterface> chain) {
  size_t n = 0;
  while (n < chain.size() && !transitivelyFeedsStore(chain[n]))
    ++n;
  return n;
}

// Hoists `chain` (feeding `loadOp`) into stride ops inserted via `builder`
// before `closure`, composing hop-by-hop from the outer-scope value that
// `loadOp.getTensor()` (and any other operand a transform reads, e.g. a
// slice's starts/stops/steps) shadows -- `outerValues` maps every capture
// block argument back to it, since the hoisted ops land outside the
// isolated closure and can't read a capture's inner shadow directly.
// Rewrites `loadOp` to read the result with the chain's outermost index.
// The hoisted tensor becomes a brand new capture; no dedup against an
// identical prior hoist here -- `DedupeCapturingClosureCaptures` folds
// duplicates afterward.
void hoistChain(LoadOp loadOp, ArrayRef<IndexTransformOpInterface> chain,
                OpBuilder &builder, CapturingFusionClosure closure,
                const IRMapping &outerValues) {
  Value outerTensor = outerValues.lookupOrDefault(loadOp.getTensor());

  Value tensor = outerTensor;
  for (IndexTransformOpInterface transform : chain)
    tensor = transform.createStridedView(builder, tensor, outerValues);
  BlockArgument newCaptureArg = closure.addCapture(tensor);

  IndexTransformOpInterface outermost = chain.back();
  loadOp.getTensorMutable().assign(newCaptureArg);
  loadOp.getIndexMutable().assign(outermost.getIndex());
}

// A fused elementwise producer shares its consumer's transformed index the
// same way a fused load does; rewire it to the chain's outermost index too,
// with no tensor/view to hoist.
void rewireElementwiseIndices(Region &body) {
  // Rewiring only reassigns each call's own index operand, so it is safe to do
  // in-walk: no op is created, erased, or moved within `body`.
  body.walk([&](CallElementwiseOp call) {
    SmallVector<IndexTransformOpInterface> chain =
        collectIndexTransformChain(call.getIndex());
    size_t prefix = hoistablePrefixLen(chain);
    if (prefix == 0)
      return; // Indexes with the loop's own index, or computes at (or
              // through) a store-index-transformed position, which stays.
    // The hoistable prefix's views are gone; rewire to the input of the last
    // hoisted op -- the loop index when the whole chain hoisted, else the first
    // store-index transform's result the pinned suffix still computes at.
    call.getIndexMutable().assign(chain[prefix - 1].getIndex());
  });
}

LogicalResult loopInvariantViewMotionInClosure(CapturingFusionClosure closure) {
  Region &body = closure.getBody();
  OpBuilder builder(closure.getOperation());
  builder.setInsertionPoint(closure.getOperation());

  // Maps each capture block argument to the outer-scope value it shadows,
  // for the hoisted stride ops (which land outside the isolated closure) to
  // read instead.
  IRMapping outerValues;
  Block::BlockArgListType captureArgs = closure.getCaptureBlockArgs();
  ValueRange captures = closure.getCaptures();
  for (auto [arg, capture] : llvm::zip_equal(captureArgs, captures))
    outerValues.map(arg, capture);

  SmallVector<LoadOp> loads;
  body.walk([&](LoadOp op) { loads.push_back(op); });

  for (LoadOp loadOp : loads) {
    SmallVector<IndexTransformOpInterface> chain =
        collectIndexTransformChain(loadOp.getIndex());
    size_t prefix = hoistablePrefixLen(chain);
    if (prefix == 0)
      continue; // Nothing hoistable: the load already indexes with the loop's
                // own index, or reads at (or through) a store-index-transformed
                // position, which stays in the closure.
    // Hoist only the load-side prefix above the first store-index transform;
    // `hoistChain` rewires the load to read the hoisted view at that
    // transform's result (e.g. a broadcast view read at the reshaped store
    // index).
    chain.truncate(prefix);
    hoistChain(loadOp, chain, builder, closure, outerValues);
  }

  rewireElementwiseIndices(body);

  // Every transform op is now unused (rewired above); erase closest-to-load
  // first so each inner hop's use of the next outer one is dropped before
  // that outer hop is considered.
  SmallVector<IndexTransformOpInterface> transforms;
  body.walk([&](IndexTransformOpInterface op) { transforms.push_back(op); });
  for (IndexTransformOpInterface transform : llvm::reverse(transforms))
    if (transform->use_empty())
      transform->erase();

  return success();
}

struct LoopInvariantViewMotionPass
    : public MAP::impl::LoopInvariantViewMotionBase<
          LoopInvariantViewMotionPass> {
  using Base::Base;

  void runOnOperation() override {
    MO::GraphOp graphOp = getOperation();

    // Walk rather than iterate the body's direct ops: a kernel can sit inside
    // control flow (e.g. mo.while), and iterating only the graph's own
    // top-level ops would miss those -- leaving an index-form view that
    // map-to-mogg cannot lower.
    WalkResult liftResult = graphOp.walk([&](CapturingFusionClosure closure) {
      if (failed(loopInvariantViewMotionInClosure(closure)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (liftResult.wasInterrupted())
      return signalPassFailure();

    // Invariant: hoisting above must leave no HOISTABLE (load-side)
    // index-transform op behind inside any capturing closure. An op that
    // transitively feeds a store legitimately stays -- it maps an index, not a
    // tensor, so there is nothing to hoist.
    WalkResult invariant = graphOp.walk([&](IndexTransformOpInterface op) {
      if (transitivelyFeedsStore(op))
        return WalkResult::advance();
      op.emitError(
          "LoopInvariantViewMotion: an index-transform op survived hoisting "
          "-- this is a bug in the pass, not in the input IR");
      return WalkResult::interrupt();
    });
    if (invariant.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

} // namespace M::MAP
