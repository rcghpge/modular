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
// runs once per kernel invocation instead of once per loop index. Runs once
// per map.iter.opaque instance, independent instances in parallel.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
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

// A store-index transform: an index-transform op (e.g. a reshape fused into an
// epilogue) that feeds a `map.store`, projecting the loop index to the store's
// target position. Unlike a load-side view it cannot be hoisted into a
// materialized `.stride` tensor -- it maps an INDEX, not a tensor -- so it (and
// any load/elementwise op that indexes THROUGH it, reading/computing at the
// transformed position) must stay inside the closure.
bool isStoreIndexTransform(IndexTransformOpInterface op) {
  return llvm::any_of(op->getUsers(),
                      [](Operation *user) { return isa<StoreOp>(user); });
}

bool chainHasStoreIndexTransform(ArrayRef<IndexTransformOpInterface> chain) {
  return llvm::any_of(chain, isStoreIndexTransform);
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
  SmallVector<CallElementwiseOp> calls;
  body.walk([&](CallElementwiseOp op) { calls.push_back(op); });
  for (CallElementwiseOp call : calls) {
    SmallVector<IndexTransformOpInterface> chain =
        collectIndexTransformChain(call.getIndex());
    if (chain.empty())
      continue; // Already indexes with the loop's own index; nothing to do.
    if (chainHasStoreIndexTransform(chain))
      continue; // Store-index transform stays; compute at the transformed pos.
    IndexTransformOpInterface outermost = chain.back();
    call.getIndexMutable().assign(outermost.getIndex());
  }
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
    if (chain.empty())
      continue; // Already indexes with the loop's own index; nothing to do.
    if (chainHasStoreIndexTransform(chain))
      continue; // Load reads at a store-index-transformed pos; leave in place.
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
    IterOpaqueOp opaque = getOperation();

    WalkResult liftResult = opaque.walk([&](CapturingFusionClosure closure) {
      if (failed(loopInvariantViewMotionInClosure(closure)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (liftResult.wasInterrupted())
      return signalPassFailure();

    // Invariant: hoisting above must leave no HOISTABLE (load-side)
    // index-transform op behind inside any capturing closure. A store-index
    // transform legitimately stays -- it maps an index, not a tensor, so there
    // is nothing to hoist.
    WalkResult invariantResult = opaque.walk([&](IndexTransformOpInterface op) {
      if (isStoreIndexTransform(op))
        return WalkResult::advance();
      op.emitError(
          "LoopInvariantViewMotion: an index-transform op survived hoisting "
          "-- this is a bug in the pass, not in the input IR");
      return WalkResult::interrupt();
    });
    if (invariantResult.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

} // namespace M::MAP
