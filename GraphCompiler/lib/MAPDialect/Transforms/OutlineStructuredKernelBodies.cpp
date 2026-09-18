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

  // Both ops keep their chain in a dedicated operand, separate from `$inputs`
  // -- `map.iter.opaque`'s own body still shadows it (unlike
  // `map.iter.foreach`'s, which never touches it), so the foreach's input
  // chain (if any) is threaded straight into the new op's own `$inputChain`.
  auto opaque = IterOpaqueOp::create(
      builder, loc, foreachOp.getResultTypes(), foreachOp.getInputs(),
      foreachOp.getInputChain(), /*fusionInfo=*/Attribute());
  // `map.iter.opaque` states its device outright, so settle the foreach's
  // type-inferred one here. A foreach whose tensors carry no annotation runs
  // on the host, which is the fallback `getDeviceOrDefault` supplies.
  opaque.setDeviceRef(MO::getDeviceOrDefault(foreachOp));
  if (Attribute fusionInfo = foreachOp.getFusionInfoAttr())
    opaque.setFusionInfoAttr(fusionInfo);
  opaque.setHasReadEffect(foreachOp.getHasReadEffect());
  opaque.setHasWriteEffect(foreachOp.getHasWriteEffect());
  opaque.setIsAsync(foreachOp.getIsAsync());

  // `map.iter.opaque` shadows only its non-chain inputs and outputs as body
  // block args; the chain rides the dedicated `$inputChain` operand and the
  // trailing chain result, never getting a block arg. Those shadowed args are
  // exactly `map.iter.foreach.body`'s captures -- the narrower
  // `CapturingFusionClosure` concept a chain is never part of, being a
  // sequencing token rather than a per-element value the body reads or writes.
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

  // Move the foreach's body block wholesale into the new op's region -- no
  // cloning. The flat foreach's own block-arg order is [inputs..., outputs...,
  // index]; `map.iter.foreach.body` wants [index, captures...] (index leads,
  // per `CapturingFusionClosure`), so only the index needs to move -- the
  // input/output args themselves are untouched (same BlockArguments, so every
  // existing use inside the body stays valid) and simply become the block
  // args shadowing `captures` above, one position later.
  //
  // The index is always the body's last argument -- computing its position
  // from foreachOp.getInputs().size() + foreachOp.getResults().size() is
  // wrong whenever one of those results is a chain: a chain is a sequencing
  // token on the op itself, with no corresponding body block argument, so it
  // doesn't contribute to the body's actual argument count.
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
