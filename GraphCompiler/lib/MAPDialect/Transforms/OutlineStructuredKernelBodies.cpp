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
// nested `map.iter.foreach.body`. `map.iter.opaque` is isolated from above, so
// its own inputs are shadowed as body block arguments; the nested body is
// *also* isolated from above (see `CapturingFusionClosure`), so those same
// opaque block args are threaded into it as `$captures`, one level further in.
void outlineForeach(IterForeachOp foreachOp) {
  Location loc = foreachOp.getLoc();
  OpBuilder builder(foreachOp);

  auto opaque = IterOpaqueOp::create(builder, loc, foreachOp.getResultTypes(),
                                     foreachOp.getInputs());
  // `map.iter.opaque` states its device outright, so settle the foreach's
  // type-inferred one here. A foreach whose tensors carry no annotation runs
  // on the host, which is the fallback `getDeviceOrDefault` supplies.
  opaque.setDeviceRef(MO::getDeviceOrDefault(foreachOp));
  if (Attribute fusionInfo = foreachOp.getFusionInfoAttr())
    opaque.setFusionInfoAttr(fusionInfo);
  opaque.setHasReadEffect(foreachOp.getHasReadEffect());
  opaque.setHasWriteEffect(foreachOp.getHasWriteEffect());

  Block *opaqueBody = builder.createBlock(&opaque.getBody());
  SmallVector<Value> captures;
  for (Value input : foreachOp.getInputs())
    captures.push_back(opaqueBody->addArgument(input.getType(), loc));
  for (Type resultType : foreachOp.getResultTypes())
    captures.push_back(opaqueBody->addArgument(resultType, loc));

  builder.setInsertionPointToStart(opaqueBody);
  auto body =
      IterForeachBodyOp::create(builder, loc, foreachOp.getShape(), captures);

  unsigned numInputs = foreachOp.getInputs().size();
  unsigned numOutputs = foreachOp.getResults().size();

  // Move the foreach's body block wholesale into the new op's region -- no
  // cloning. The flat foreach's own block-arg order is [inputs..., outputs...,
  // index]; `map.iter.foreach.body` wants [index, captures...] (index leads,
  // per `CapturingFusionClosure`), so only the index needs to move -- the
  // input/output args themselves are untouched (same BlockArguments, so every
  // existing use inside the body stays valid) and simply become the block
  // args shadowing `captures` above, one position later.
  body.getBody().takeBody(foreachOp.getBody());
  Block &movedBody = body.getBody().front();
  BlockArgument oldIndexArg = movedBody.getArgument(numInputs + numOutputs);
  BlockArgument newIndexArg =
      movedBody.insertArgument(0u, oldIndexArg.getType(), loc);
  oldIndexArg.replaceAllUsesWith(newIndexArg);
  movedBody.eraseArgument(numInputs + numOutputs + 1);

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
