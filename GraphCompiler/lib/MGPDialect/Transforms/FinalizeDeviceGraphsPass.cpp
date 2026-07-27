//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "llvm/ADT/SmallVector.h"

using namespace M;
using namespace MGP;
using namespace mlir;

namespace {

using DeviceContextValue = mlir::TypedValue<DeviceContextType>;

DeviceRefAttr findGraphDevice(ValueRange values) {
  for (Value value : values) {
    if (auto deviceType =
            dyn_cast<MGPHasDeviceTypeInterface>(value.getType())) {
      return deviceType.getDevice();
    }
  }

  return {};
}

// device graphs have no chain concept: a host-side sequencing chain is
// neither a real graph input nor output (in the staged body the chain inputs
// are dead pass-throughs and the chain outputs come from host-side
// `mogg.async.ready`). Drop chain-typed values from the graph's I/O; the
// ordering chain is produced fresh by the `mgp.device_graph.execute` op
// below.
bool isChain(Value v) { return isa<MEF::ChainType>(v.getType()); }

SmallVector<Value> nonChainValues(ValueRange v) {
  auto isNonChainValue = [](Value v) { return !isChain(v); };

  return llvm::filter_to_vector(v, isNonChainValue);
}

/// Lower a single `mgp.device_graph.staging` op into a
/// `mgp.device_graph.create` / `mgp.device_graph.execute` pair, rewiring
/// `mef.output` and erasing the staging op.
void finalizeStagingOp(DeviceGraphStagingOp stagingOp, ModelOp model) {
  Block &stagingBody = stagingOp.getBodyBlock();
  Value stagingBuilderArg = stagingBody.getArgument(0);
  auto stagingInputArgs = stagingBody.getArguments().drop_front(1);

  // The escaping (non-chain) values the staged body yields become the graph's
  // outputs.
  SmallVector<Value> graphOutputs =
      nonChainValues(stagingOp.getYieldOp().getOperands());

  // create/execute draw the model inputs from the staging op and the context /
  // device context from the execute block.
  SmallVector<Value> inputs = nonChainValues(stagingOp.getInputs());

  Value theContext = model.getExecuteContext();

  // The device graph runs on the device its buffers/tensors live on. The
  // model's execute devices list the host (CPU) context first, and the host
  // context cannot build a device graph (`createGraphBuilder` is unsupported
  // there), so `.front()` is wrong for accelerator graphs.
  // Determine the graph's device from its I/O and select the matching execute
  // device context. (First cut: a single device per model, so one device
  // covers the whole graph.)
  DeviceRefAttr graphDevice = findGraphDevice(graphOutputs);
  if (!graphDevice)
    findGraphDevice(inputs);

  DeviceContextValue deviceContext;
  for (auto [deviceRef, execDevice] :
       llvm::zip(model.getDevices(), model.getExecuteDevices())) {
    if (deviceRef == graphDevice) {
      deviceContext = cast<DeviceContextValue>(execDevice);
      break;
    }
  }
  assert(deviceContext &&
         "no execute device context matches the device graph's device");

  IRRewriter rewriter(stagingOp);
  Location loc = stagingOp.getLoc();
  MLIRContext *ctx = rewriter.getContext();

  // The `mgp.device_graph.create` op is compiled like any other kernel by
  // wrapping it in a `mogg.generic.compiled.region`.
  auto graphType = DeviceGraphType::get(ctx, inputs, graphOutputs);

  SmallVector<Value> regionOperands{theContext, deviceContext};
  llvm::append_range(regionOperands, inputs);

  SmallVector<InOutSignatureAttr::InOutSemantics> operandSemantics(
      regionOperands.size(), InOutSignatureAttr::InOutSemantics::kNone);

  auto regionOp = MOGG::GenericCompiledRegionOp::create(
      rewriter, loc, /*results=*/TypeRange{graphType},
      /*operands=*/regionOperands,
      /*outputParamDecls=*/KGEN::ParamDeclArrayAttr::get(ctx, {}),
      /*inOutSignature=*/InOutSignatureAttr::get(ctx, operandSemantics),
      /*fusionInfo=*/
      MOGG::FusionInfoAttr::get(
          ctx, MOGG::FusionLeafAttr::get(
                   ctx, StringAttr::get(ctx, "mgp.device_graph.create"),
                   /*loc=*/{}, /*signature=*/{})),
      /*isAsync=*/rewriter.getBoolAttr(false),
      /*constantResources=*/stagingOp.getConstantResources(),
      /*canFuseIntoOther=*/rewriter.getBoolAttr(false),
      /*canBeFusedInto=*/rewriter.getBoolAttr(false),
      /*canDuplicate=*/rewriter.getBoolAttr(false));

  regionOp.setDeviceRef(deviceContext.getType().getDevice());

  // The region's body block args are `!mogg.async<T>` wrappers around the
  // operands; unpack them to plain `T` at the top.
  Block &regionBody = regionOp.getBody().emplaceBlock();
  for (Value operand : regionOperands)
    regionBody.addArgument(MOGG::AsyncType::get(operand.getType()), loc);
  rewriter.setInsertionPointToStart(&regionBody);

  auto unpackAsyncValue = [&](Value val) -> Value {
    return MOGG::AsyncUnpackOp::create(rewriter, loc, val);
  };

  SmallVector<Value> unpacked =
      llvm::map_to_vector(regionBody.getArguments(), unpackAsyncValue);

  Value contextUnpacked = unpacked[0];
  Value deviceContextUnpacked = unpacked[1];
  ValueRange inputsUnpacked = ArrayRef<Value>(unpacked).drop_front(2);

  // Build the stateless create op inside the region body, moving the staged
  // body into its build body. The create builder makes the body block with
  // [builder, context, inputs...] args; the callback splices the staged ops in,
  // rebinds the staged block args, and yields the graph outputs (from which the
  // result graph signature is inferred). The `context` block arg is build-time
  // only (for the body's `get_cached` reads); the staged body in the first cut
  // does not reference it, so it is left unused here.
  auto createOp = DeviceGraphCreateOp::create(
      rewriter, loc, contextUnpacked, deviceContextUnpacked, inputsUnpacked,
      [&](OpBuilder &bodyBuilder, Location bodyLoc, Value graphBuilder,
          Value /*contextArg*/, ValueRange inputArgs) {
        Block *createBody = bodyBuilder.getInsertionBlock();
        createBody->getOperations().splice(
            createBody->end(), stagingBody.getOperations(), stagingBody.begin(),
            std::prev(stagingBody.end()));
        stagingBuilderArg.replaceAllUsesWith(graphBuilder);
        // Rebind the staged input block args to the create body's input args.
        // Chain inputs were filtered out of the graph signature, so skip them
        // (they are unused inside the staged body).
        unsigned createArgIdx = 0;
        for (auto [stagingArg, stagingInput] :
             llvm::zip_equal(stagingInputArgs, stagingOp.getInputs())) {
          if (isChain(stagingInput))
            continue;
          stagingArg.replaceAllUsesWith(inputArgs[createArgIdx++]);
        }
        DeviceGraphYieldOp::create(bodyBuilder, bodyLoc, graphOutputs);

        // We're transitioning to an SSACFG based operation, so we need to order
        // the operations in the body to enforce dominance requirements.
        mlir::sortTopologically(createBody);
      });

  // Pack the built graph back into async and yield it from the region.
  rewriter.setInsertionPointAfter(createOp);
  Value packedGraph =
      MOGG::AsyncPackOp::create(rewriter, loc, createOp.getOutGraph());
  MOGG::OutputOp::create(rewriter, loc, ValueRange{packedGraph});

  // Replay (outside the region): feed the same (non-chain) inputs, surface the
  // graph's value outputs as results, and produce a fresh completion chain that
  // host-side ordering hangs off of. The replay is ordered after the build by
  // the graph value. Reset the insertion point outside the region (it was left
  // in the region body) so the execute op replaces the staging op in the
  // execute block.
  rewriter.setInsertionPoint(stagingOp);
  auto executeOp = DeviceGraphExecuteOp::create(
      rewriter, loc, /*chain=*/MEF::ChainType::get(ctx),
      /*outputs=*/TypeRange(graphOutputs),
      /*graph=*/regionOp.getResult(0),
      /*deviceContext=*/deviceContext,
      /*inputs=*/inputs);

  // Rewire the staging op's results: chain results (host ordering) map to the
  // execute completion chain; value results map to the execute value outputs in
  // yield order.
  Value executeChain = executeOp.getChain();
  unsigned outputIdx = 0;
  for (Value stagingResult : stagingOp.getResults()) {
    if (isChain(stagingResult))
      stagingResult.replaceAllUsesWith(executeChain);
    else
      stagingResult.replaceAllUsesWith(executeOp.getOutputs()[outputIdx++]);
  }
  rewriter.eraseOp(stagingOp);
}

} // namespace

namespace M::MGP {

#define GEN_PASS_DEF_FINALIZEDEVICEGRAPHSPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

class FinalizeDeviceGraphsPass
    : public impl::FinalizeDeviceGraphsPassBase<FinalizeDeviceGraphsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModelOp model = getOperation();
    if (!model.getIsDeviceGraph())
      return;

    // One staging op per device today; each is lowered independently (the
    // create/execute pair carries no chain, so there is nothing to thread).
    for (DeviceGraphStagingOp stagingOp : llvm::to_vector(
             model.getExecuteBlock()->getOps<DeviceGraphStagingOp>()))
      finalizeStagingOp(stagingOp, model);
  }
};

} // namespace
} // namespace M::MGP
