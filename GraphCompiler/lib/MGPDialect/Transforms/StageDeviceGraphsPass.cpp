//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "GraphCompiler/MGPDialect/Transforms/Utils.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/IRMapping.h"

using namespace M::MGP;
using namespace M::MOGG;
using namespace mlir;

namespace {

/// Rewrite each `mgp.guard` in the model's execute block into a forwarding
/// `mogg.generic.compiled.region`.
///
/// `mgp.guard` lowers to a runtime primitive, not through the EmitMojo path, so
/// device-graph staging never turns it into a `mogg.generic.compiled.region`
/// and leaves it in the execute block. When a guard sits in the middle of the
/// device-graph computation -- its chain comes from one staged region and its
/// forwarded result feeds another -- that stranded guard forms a use-def cycle
/// with the staging op (a staging result feeds the guard, whose result feeds
/// back in as a staging input). Turning the guard into a compiled region up
/// front lets it be staged like any other region, breaking the cycle.
///
/// The region forwards its non-chain inputs straight to its results; the
/// leading chain operand only orders the region (its block arg is intentionally
/// unused), preserving the guard's "gate these values on this chain" semantics.
void lowerGuardsToCompiledRegions(MLIRContext *ctx, ModelOp model) {
  auto guards = llvm::to_vector(model.getExecuteBlock()->getOps<GuardOp>());
  for (GuardOp guard : guards) {
    OpBuilder builder(guard);
    Location loc = guard.getLoc();

    ValueRange operands = guard->getOperands();

    // The forwarded values carry the device; the leading chain operand does
    // not. All in-device-graph guards forward at least one device value.
    M::DeviceRefAttr device;
    for (Value result : guard.getResults())
      if (auto deviceType =
              dyn_cast<MGPHasDeviceTypeInterface>(result.getType())) {
        device = deviceType.getDevice();
        break;
      }

    SmallVector<M::InOutSignatureAttr::InOutSemantics> operandSemantics(
        operands.size(), M::InOutSignatureAttr::InOutSemantics::kNone);

    auto regionOp = GenericCompiledRegionOp::create(
        builder, loc, /*results=*/guard.getResultTypes(),
        /*operands=*/operands,
        /*outputParamDecls=*/KGEN::ParamDeclArrayAttr::get(ctx, {}),
        /*inOutSignature=*/M::InOutSignatureAttr::get(ctx, operandSemantics),
        /*fusionInfo=*/
        FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(guard)),
        /*isAsync=*/builder.getBoolAttr(false),
        /*constantResources=*/builder.getArrayAttr({}),
        /*canFuseIntoOther=*/builder.getBoolAttr(false),
        /*canBeFusedInto=*/builder.getBoolAttr(false),
        /*canDuplicate=*/builder.getBoolAttr(false));
    if (device)
      regionOp.setDeviceRef(device);

    // The body's block args are `!mogg.async<T>` wrappers around the operands.
    // Unpack then re-pack each forwarded input (the operands after the leading
    // chain) and yield them; the chain block arg is left unused -- it only
    // establishes the ordering dependency on the region.
    Block &body = regionOp.getBody().emplaceBlock();
    for (Value operand : operands)
      body.addArgument(AsyncType::get(operand.getType()), loc);
    builder.setInsertionPointToStart(&body);

    OutputOp::create(builder, loc, body.getArguments().drop_front());

    guard->replaceAllUsesWith(regionOp.getResults());
    guard.erase();
  }
}

void stageDeviceGraphOps(MLIRContext *ctx, ModelOp model) {
  // Fold interleaved `mgp.guard` ops into compiled regions so they stage with
  // the rest of the computation (otherwise they strand and form a cycle).
  lowerGuardsToCompiledRegions(ctx, model);

  Block *executeBlock = model.getExecuteBlock();

  /// Collect the ops that will move into the staging body: every
  /// `mogg.generic.compiled.region`, plus any `mgp.device_graph.stable.create`
  /// whose operand is produced by an op already in the staged set. Walking the
  /// block in order preserves the original scheduling.
  llvm::DenseSet<Operation *> stagedOpSet;
  for (GenericCompiledRegionOp region :
       executeBlock->getOps<GenericCompiledRegionOp>())
    stagedOpSet.insert(region);

  /// Stable-create ops that promote a staged producer result belong inside the
  /// device-graph staging body; keeping them outside would create a dataflow
  /// cycle with the staging op.
  for (DeviceGraphStableCreateOp stableCreate :
       executeBlock->getOps<DeviceGraphStableCreateOp>()) {
    Operation *producer = stableCreate.getTensor().getDefiningOp();
    if (producer && stagedOpSet.contains(producer))
      stagedOpSet.insert(stableCreate);
  }

  SmallVector<Operation *> stagedOps;
  for (Operation &op : *executeBlock)
    if (stagedOpSet.contains(&op))
      stagedOps.push_back(&op);

  if (stagedOps.empty())
    return;

  /// Free values: operands of staged ops that are defined outside the staged
  /// set. The staging op is `IsolatedFromAbove`, so these become its `inputs`
  /// operands and enter the body as bare-`T` block arguments.
  llvm::SetVector<Value> freeValues;
  for (Operation *op : stagedOps) {
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def || !stagedOpSet.contains(def))
        freeValues.insert(operand);
    }
  }

  auto inputs = freeValues.takeVector();

  /// The set of uses of values produced inside our new region op by operations
  /// that will remain outside the region op.
  llvm::SetVector<OpOperand *> escapingUses;
  for (Operation *op : stagedOps) {
    for (OpResult result : op->getResults()) {
      for (OpOperand &use : result.getUses()) {
        if (!stagedOpSet.contains(use.getOwner()))
          escapingUses.insert(&use);
      }
    }
  }

  auto locations = llvm::map_to_vector(
      stagedOps, [](Operation *op) { return op->getLoc(); });

  Location newLoc = FusedLoc::get(ctx, locations);

  auto buildGraphStagingBody = [&](OpBuilder &builder, Location loc, Value,
                                   ValueRange inputArgs) {
    /// Map each free value to the bare-`T` block arg representing it. The async
    /// pack/unpack mediation a dissolved `mogg.generic.compiled.region` needs
    /// is injected later, by `mgp-compute-device-graph-dependencies`.
    IRMapping freeMap;
    freeMap.map(inputs, inputArgs);

    /// Move all staged ops into the body (preserving order).
    for (Operation *op : stagedOps)
      op->moveBefore(builder.getBlock(), builder.getBlock()->end());

    /// Rewire each staged op's operand that referenced a free value to the
    /// corresponding in-body block arg.
    for (Operation *op : stagedOps)
      for (OpOperand &operand : op->getOpOperands())
        if (Value mapped = freeMap.lookupOrNull(operand.get()))
          operand.set(mapped);

    builder.setInsertionPointToEnd(builder.getBlock());

    /// Yield the escaping values directly (bare `T`); one operand per escaping
    /// use, so repeated uses appear multiple times (cleaned by
    /// canonicalization).
    auto yieldValues = llvm::map_to_vector(
        escapingUses, [](OpOperand *use) { return use->get(); });

    DeviceGraphStagingYieldOp::create(builder, newLoc, yieldValues);
  };

  /// Anchor the insertion point on the execute terminator rather than the first
  /// staged op: the staging-body callback moves every staged op (including the
  /// first) into the new op's body, so anchoring on an op would leave the
  /// builder's restored insertion point pointing at a relocated op and the
  /// staging op would be inserted detached from the execute block. The
  /// terminator does not move, and is the correct spot anyway (mef.output
  /// consumes the staging results).
  OpBuilder builder(model.getExecuteBlock()->getTerminator());

  auto stagingOp = DeviceGraphStagingOp::create(builder, newLoc, inputs,
                                                buildGraphStagingBody);

  for (auto [oldUse, newValue] :
       llvm::zip_equal(escapingUses, stagingOp.getResults()))
    oldUse->set(newValue);
}

} // namespace

namespace M::MGP {

#define GEN_PASS_DEF_STAGEDEVICEGRAPHSPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

class StageDeviceGraphsPass
    : public impl::StageDeviceGraphsPassBase<StageDeviceGraphsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModelOp model = getOperation();

    if (model.getIsDeviceGraph())
      stageDeviceGraphOps(&getContext(), model);
  }
};

} // namespace
} // namespace M::MGP
