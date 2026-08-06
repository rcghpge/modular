//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Wraps structured MAP loops in `map.iter.opaque` regions so sequential setup
// can live outside the structured loop body.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "mlir/IR/Builders.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_OUTLINESTRUCTUREDKERNELBODIES
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// Wraps `foreachOp` in a `map.iter.opaque`, moving its shape and body into a
// nested `map.iter.foreach.body`. The wrapper's own operands stay exactly
// `foreachOp`'s inputs, unrenamed; the nested body reaches them -- and the
// wrapper's output block args -- by plain SSA dominance, since neither the
// wrapper nor the nested body is isolated from above.
void outlineForeach(IterForeachOp foreachOp) {
  Location loc = foreachOp.getLoc();
  OpBuilder builder(foreachOp);

  auto opaque = IterOpaqueOp::create(builder, loc, foreachOp.getResultTypes(),
                                     foreachOp.getInputs());
  if (Attribute fusionInfo = foreachOp.getFusionInfoAttr())
    opaque.setFusionInfoAttr(fusionInfo);

  Block *opaqueBody = builder.createBlock(&opaque.getBody());
  SmallVector<Value> outputBlockArgs;
  for (Type resultType : foreachOp.getResultTypes())
    outputBlockArgs.push_back(opaqueBody->addArgument(resultType, loc));

  builder.setInsertionPointToStart(opaqueBody);
  auto body = IterForeachBodyOp::create(builder, loc, foreachOp.getShape());

  unsigned numInputs = foreachOp.getInputs().size();
  unsigned numOutputs = foreachOp.getResults().size();

  // Move the foreach's body block wholesale into the new op's region -- no
  // cloning -- then drop its input/output block args in favor of direct
  // references to the values they'd otherwise have shadowed: `foreachOp`'s
  // own input operands (now `opaque`'s inputs, dominating unchanged) and the
  // new opaque body's own output block args. What's left is exactly the
  // index block arg `map.iter.foreach.body` expects.
  body.getBody().takeBody(foreachOp.getBody());
  Block &movedBody = body.getBody().front();
  for (unsigned i = 0; i < numInputs; ++i)
    movedBody.getArgument(i).replaceAllUsesWith(foreachOp.getInputs()[i]);
  for (unsigned i = 0; i < numOutputs; ++i)
    movedBody.getArgument(numInputs + i).replaceAllUsesWith(outputBlockArgs[i]);
  movedBody.eraseArguments(0, numInputs + numOutputs);

  foreachOp.replaceAllUsesWith(opaque.getResults());
  foreachOp.erase();
}

struct OutlineStructuredKernelBodiesPass
    : public MAP::impl::OutlineStructuredKernelBodiesBase<
          OutlineStructuredKernelBodiesPass> {
  using Base::Base;

  void runOnOperation() override {
    SmallVector<IterForeachOp> worklist;
    getOperation()->walk([&](IterForeachOp op) {
      worklist.push_back(op);
      return WalkResult::skip();
    });

    for (IterForeachOp op : worklist)
      outlineForeach(op);
  }
};

} // namespace

} // namespace M::MAP
