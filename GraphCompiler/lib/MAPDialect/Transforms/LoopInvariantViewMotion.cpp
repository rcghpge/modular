//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Hoists chains of index-transform ops (ops implementing
// MAP::IndexTransformOpInterface, e.g. map.index.slice) out of a
// map.iter.foreach.body into a chain of "stride" ops (e.g.
// map.index.slice.stride) in the enclosing map.iter.opaque, so the transform
// runs once per kernel invocation instead of once per loop index. Runs once
// per map.iter.opaque instance, independent instances in parallel.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_LOOPINVARIANTVIEWMOTION
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// Walks backward from `index` through a chain of index-transform ops,
// collecting each hop closest-to-load first, and stops at the first value
// that isn't a recognized index-transform op's result (typically the loop's
// own `map.iter.foreach.body` index block argument). Generic over which
// concrete transform kind is involved (today: only `map.index.slice`) via
// `IndexTransformOpInterface`.
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

// Keyed by (the chain's innermost transform op, the tensor being loaded): the
// same index-transform result could in principle feed loads of *different*
// tensors (an index-transform op carries no tensor operand of its own -- the
// tensor lives on the map.load), so keying by the transform op alone would
// incorrectly hand a second, unrelated tensor the first tensor's hoisted
// view. Keying by the pair still dedups the common case, an index shared by
// multiple loads of the *same* tensor. The innermost op alone identifies the
// whole chain, since the rest is deterministically reachable by walking its
// `getIndex()` operand.
using HoistKey = std::pair<Operation *, Value>;

// Hoists the index-transform `chain` feeding `loadOp` into a chain of stride
// ops inserted via `builder` (positioned before the enclosing
// map.iter.foreach.body), and rewrites `loadOp` to read the fully-hoisted
// tensor with the chain's outermost (untransformed) index directly. `chain`
// is ordered closest-to-load first, so composing hop-by-hop starting from
// `loadOp.getTensor()` builds up the strided view innermost-to-outermost --
// each hop's hoisted tensor becomes the next hop's input, ending in a tensor
// a load can address with the outermost hop's own (untransformed) index.
// `hoisted` caches by (innermost transform op, tensor) so multiple loads of
// the same tensor sharing the exact same chain (SSA-identical, not just
// structurally similar) hoist once.
LogicalResult hoistChain(LoadOp loadOp,
                         ArrayRef<IndexTransformOpInterface> chain,
                         llvm::DenseMap<HoistKey, Value> &hoisted,
                         OpBuilder &builder) {
  IndexTransformOpInterface innermost = chain.front();
  Value &strided = hoisted[{innermost.getOperation(), loadOp.getTensor()}];
  if (!strided) {
    Value tensor = loadOp.getTensor();
    for (IndexTransformOpInterface transform : chain)
      tensor = transform.createStridedView(builder, tensor);
    strided = tensor;
  }

  IndexTransformOpInterface outermost = chain.back();
  Value outerIndex = outermost.getIndex();
  loadOp.getTensorMutable().assign(strided);
  loadOp.getIndexMutable().assign(outerIndex);

  // Each hop may now be dead (this was its only load); other loads may still
  // share a hop via `hoisted`, so only erase once unused. Closest-to-load
  // first, so an inner hop's own use of the next outer hop's result is
  // dropped before that outer hop is considered.
  for (IndexTransformOpInterface transform : chain)
    if (transform->use_empty())
      transform->erase();

  return success();
}

LogicalResult loopInvariantViewMotionInBody(IterForeachBodyOp body) {
  OpBuilder builder(body);
  builder.setInsertionPoint(body);

  llvm::DenseMap<HoistKey, Value> hoisted;
  SmallVector<LoadOp> loads;
  body.walk([&](LoadOp op) { loads.push_back(op); });

  for (LoadOp loadOp : loads) {
    SmallVector<IndexTransformOpInterface> chain =
        collectIndexTransformChain(loadOp.getIndex());
    if (chain.empty())
      continue; // Already indexes with the loop's own index; nothing to do.
    if (failed(hoistChain(loadOp, chain, hoisted, builder)))
      return failure();
  }

  return success();
}

struct LoopInvariantViewMotionPass
    : public MAP::impl::LoopInvariantViewMotionBase<
          LoopInvariantViewMotionPass> {
  using Base::Base;

  void runOnOperation() override {
    IterOpaqueOp opaque = getOperation();

    WalkResult liftResult = opaque.walk([&](IterForeachBodyOp body) {
      if (failed(loopInvariantViewMotionInBody(body)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (liftResult.wasInterrupted())
      return signalPassFailure();

    // Invariant: hoisting above must leave no index-transform op behind
    // inside any structured loop body.
    WalkResult invariantResult = opaque.walk([&](IndexTransformOpInterface op) {
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
