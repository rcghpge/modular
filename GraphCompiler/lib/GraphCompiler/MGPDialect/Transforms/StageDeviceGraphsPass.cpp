//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "GraphCompiler/GraphCompiler/MGPDialect/Transforms/Utils.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "mlir/IR/IRMapping.h"

using namespace M::MGP;
using namespace M::MOGG;
using namespace mlir;

namespace {

void stageDeviceGraphOps(MLIRContext *ctx, ModelOp model) {
  auto ops = llvm::to_vector(
      model.getExecuteBlock()->getOps<GenericCompiledRegionOp>());

  if (ops.empty())
    return;

  llvm::SetVector<Operation *> bodyOps;
  bodyOps.insert(ops.begin(), ops.end());

  /// Free values: region operands defined outside the staged set. The staging
  /// op is `IsolatedFromAbove`, so these become its `inputs` operands and enter
  /// the body as `!mogg.async<T>` block arguments. A compiled region is itself
  /// isolated, so it can only reference outer values through its operands --
  /// remapping those is all that is needed.
  llvm::SetVector<Value> freeValues;
  for (GenericCompiledRegionOp op : ops) {
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def || !bodyOps.contains(def))
        freeValues.insert(operand);
    }
  }

  auto inputs = freeValues.takeVector();

  /// The set of uses of values produced inside our new region op by operations
  /// that will remain outside the region op.
  llvm::SetVector<OpOperand *> escapingUses;
  for (GenericCompiledRegionOp op : ops) {
    for (OpResult result : op->getResults()) {
      for (OpOperand &use : result.getUses()) {
        if (!bodyOps.contains(use.getOwner()))
          escapingUses.insert(&use);
      }
    }
  }

  auto locations = llvm::map_to_vector(
      ops, [](GenericCompiledRegionOp op) { return op.getLoc(); });

  Location newLoc = FusedLoc::get(ctx, locations);

  auto buildGraphStagingBody = [&](OpBuilder &builder, Location loc, Value,
                                   ValueRange asyncInputs) {
    /// Unpack each async input block arg to its plain type and map the free
    /// value it represents to the unpacked value.
    IRMapping freeMap;
    for (auto [freeValue, asyncArg] : llvm::zip_equal(inputs, asyncInputs))
      freeMap.map(freeValue, AsyncUnpackOp::create(
                                 builder, loc, freeValue.getType(), asyncArg)
                                 .getOutput());

    /// Move all regions into the body (preserving order).
    for (GenericCompiledRegionOp op : ops)
      op->moveBefore(builder.getBlock(), builder.getBlock()->end());

    /// Rewire each region operand that referenced a free value to the unpacked
    /// in-body value.
    for (GenericCompiledRegionOp op : ops)
      for (OpOperand &operand : op->getOpOperands())
        if (Value unpacked = freeMap.lookupOrNull(operand.get()))
          operand.set(unpacked);

    builder.setInsertionPointToEnd(builder.getBlock());

    /// Pack each distinct escaping value back into async for the yield (reused
    /// across repeated uses; duplicates are cleaned by canonicalization).
    DenseMap<Value, Value> packed;
    auto yieldValues =
        llvm::map_to_vector(escapingUses, [&](OpOperand *use) -> Value {
          Value escaping = use->get();
          Value &slot = packed[escaping];
          if (!slot)
            slot = AsyncPackOp::create(builder, loc, escaping);
          return slot;
        });

    DeviceGraphStagingYieldOp::create(builder, newLoc, yieldValues);
  };

  /// Anchor the insertion point on the execute terminator rather than the first
  /// compiled region: the staging-body callback moves every region (including
  /// the first) into the new op's body, so anchoring on a region would leave
  /// the builder's restored insertion point pointing at a relocated op and the
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
#include "GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

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
