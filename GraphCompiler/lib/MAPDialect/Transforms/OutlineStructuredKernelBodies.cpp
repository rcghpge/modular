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
#include "GraphCompiler/MODialect/MODevices.h"
#include "mlir/IR/Builders.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_OUTLINESTRUCTUREDKERNELBODIES
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// Wraps `foreachOp` in a `map.iter.opaque`, moving its shape and body into a
// nested `map.iter.foreach.body`. Both are isolated from above, so the opaque's
// block args are threaded into the nested body as `$captures`.
void outlineForeach(IterForeachOp foreachOp) {
  Location loc = foreachOp.getLoc();
  OpBuilder builder(foreachOp);

  auto opaque = IterOpaqueOp::create(
      builder, loc, foreachOp.getResultTypes(), foreachOp.getInputs(),
      foreachOp.getInputChain(), /*fusionInfo=*/Attribute());
  // `map.iter.opaque` states its device explicitly; an unannotated foreach
  // runs on the host, the `getDeviceOrDefault` fallback.
  opaque.setDeviceRef(MO::getDeviceOrDefault(foreachOp));
  if (Attribute fusionInfo = foreachOp.getFusionInfoAttr())
    opaque.setFusionInfoAttr(fusionInfo);
  opaque.setHasReadEffect(foreachOp.getHasReadEffect());
  opaque.setHasWriteEffect(foreachOp.getHasWriteEffect());
  opaque.setDispatch(foreachOp.getDispatch());

  // Chains get no block arg; the non-chain inputs and outputs are exactly the
  // `map.iter.foreach.body`'s captures.
  Block *opaqueBody = builder.createBlock(&opaque.getBody());
  SmallVector<Value> bodyCaptures;
  for (Value input : foreachOp.getInputs())
    bodyCaptures.push_back(opaqueBody->addArgument(input.getType(), loc));
  for (Type resultType : foreachOp.getResultTypes())
    if (!isa<MO::ChainType>(resultType))
      bodyCaptures.push_back(opaqueBody->addArgument(resultType, loc));

  builder.setInsertionPointToStart(opaqueBody);
  auto body = IterForeachBodyOp::create(builder, loc, foreachOp.getShape(),
                                        bodyCaptures);

  // Move the body without cloning. The foreach's block args are [inputs...,
  // outputs..., index] and `map.iter.foreach.body` wants [index, captures...],
  // so only the index moves. The index is the last argument; deriving its
  // position from the result count is wrong when a result is a chain, which
  // has no block arg.
  body.getBody().takeBody(foreachOp.getBody());
  Block &movedBody = body.getBody().front();
  unsigned oldIndexPos = movedBody.getNumArguments() - 1;
  BlockArgument oldIndexArg = movedBody.getArgument(oldIndexPos);
  BlockArgument newIndexArg =
      movedBody.insertArgument(0u, oldIndexArg.getType(), loc);
  oldIndexArg.replaceAllUsesWith(newIndexArg);
  movedBody.eraseArgument(oldIndexPos + 1);

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
