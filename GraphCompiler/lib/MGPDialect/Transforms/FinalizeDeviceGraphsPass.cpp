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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
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
  return llvm::filter_to_vector(v, [](Value v) { return !isChain(v); });
}

/// Replace the builder -> device-context bridges that survive synthesis with
/// `mgp.device_graph.device_context` ops.
///
/// `mgp-compute-device-graph-dependencies` re-sources an in-body device context
/// from the recording builder through a `builtin.unrealized_conversion_cast`,
/// expecting `mgp-generate-build-graph-stubs` to rewrite the consumer to take
/// the builder directly -- as it does for `mgp.buffer.alloc` -- so the round
/// trip folds away. A consumer that genuinely needs a device context (e.g.
/// `mgp.buffer.bulk_slice`, which only does host-side sub-buffer math) leaves
/// its cast live, and an unrealized cast cannot be emitted to Mojo. Fold the
/// round trips first: converting one of a pair early would strand its inverse.
LogicalResult resolveBuilderCasts(Region &region) {
  SmallVector<UnrealizedConversionCastOp> casts;
  region.walk([&](UnrealizedConversionCastOp cast) { casts.push_back(cast); });

  SmallVector<UnrealizedConversionCastOp> remaining;
  mlir::reconcileUnrealizedCasts(casts, &remaining);

  for (UnrealizedConversionCastOp cast : remaining) {
    if (cast.getNumOperands() != 1 || cast.getNumResults() != 1 ||
        !isa<DeviceGraphBuilderType>(cast.getOperand(0).getType()))
      return cast.emitError()
             << "device-graph finalization: cannot resolve this cast; only a "
                "`!mgp.device_graph.builder` to `!mgp.device_context` bridge "
                "has a device-graph equivalent";

    auto deviceContextType =
        dyn_cast<DeviceContextType>(cast.getResult(0).getType());
    if (!deviceContextType)
      return cast.emitError()
             << "device-graph finalization: cannot resolve a cast from "
                "`!mgp.device_graph.builder` to "
             << cast.getResult(0).getType();

    // Work dispatched against the backing context runs eagerly while the graph
    // is being built instead of being recorded, so only ops audited to use the
    // context purely for host-side buffer views may have it. The list is an
    // allow-list rather than an effect-freedom test on purpose: `mgp.sync` is
    // `Pure` yet records a device event, so effect-freedom does not imply
    // "enqueues nothing". An op that must record needs either a builder-native
    // form (as `mgp.device_graph.alloc` is to `mgp.buffer.alloc`) or a
    // `mgp.device_graph.recording_context`.
    for (Operation *user : cast->getUsers())
      if (!isa<BufferSliceOp, BufferBulkSliceOp, TensorSliceOp>(user))
        return cast.emitError()
               << "device-graph finalization: '" << user->getName()
               << "' is not known to be safe against the graph builder's "
                  "backing device context (it may dispatch work, which would "
                  "run eagerly at build time instead of being recorded)";

    Value graphBuilder = cast.getOperand(0);
    IRRewriter rewriter(cast);
    rewriter.replaceOpWithNewOp<DeviceGraphDeviceContextOp>(
        cast, deviceContextType, graphBuilder);
  }

  return mlir::success();
}

/// Build a `mogg.generic.compiled.region` around a device-graph build callback.
///
/// `operands` are the live values the region needs from the outside. The region
/// body receives them as `!mogg.async<T>` block arguments, unpacks them, and
/// passes the unpacked values to `bodyBuilder`. The values returned by
/// `bodyBuilder` are packed back into async and yielded.
static FailureOr<MOGG::GenericCompiledRegionOp> wrapDeviceGraphBuildRegion(
    IRRewriter &rewriter, Location loc, DeviceRefAttr deviceRef,
    TypeRange resultTypes, ValueRange operands, StringRef fusionName,
    ArrayAttr constantResources,
    llvm::function_ref<FailureOr<SmallVector<Value>>(OpBuilder &, Location,
                                                     ValueRange)>
        bodyBuilder) {
  MLIRContext *ctx = rewriter.getContext();

  SmallVector<InOutSignatureAttr::InOutSemantics> operandSemantics(
      operands.size(), InOutSignatureAttr::InOutSemantics::kNone);

  auto regionOp = MOGG::GenericCompiledRegionOp::create(
      rewriter, loc, resultTypes, operands,
      KGEN::ParamDeclArrayAttr::get(ctx, {}),
      InOutSignatureAttr::get(ctx, operandSemantics),
      MOGG::FusionInfoAttr::get(
          ctx, MOGG::FusionLeafAttr::get(ctx, StringAttr::get(ctx, fusionName),
                                         /*loc=*/{}, /*signature=*/{})),
      rewriter.getBoolAttr(false), constantResources,
      rewriter.getBoolAttr(false), rewriter.getBoolAttr(false),
      rewriter.getBoolAttr(false));
  regionOp.setDeviceRef(deviceRef);

  Block &regionBody = regionOp.getBody().emplaceBlock();
  for (Value operand : operands)
    regionBody.addArgument(MOGG::AsyncType::get(operand.getType()), loc);
  rewriter.setInsertionPointToStart(&regionBody);

  auto unpacked =
      llvm::map_to_vector(regionBody.getArguments(), [&](Value arg) -> Value {
        return MOGG::AsyncUnpackOp::create(rewriter, loc, arg);
      });

  FailureOr<SmallVector<Value>> outputs = bodyBuilder(rewriter, loc, unpacked);
  if (failed(outputs))
    return mlir::failure();

  auto packed = llvm::map_to_vector(*outputs, [&](Value out) -> Value {
    return MOGG::AsyncPackOp::create(rewriter, loc, out);
  });

  MOGG::OutputOp::create(rewriter, loc, packed);

  return regionOp;
}

/// Lower a single `mgp.device_graph.staging` op into a
/// `mgp.device_graph.create` / `mgp.device_graph.execute` pair, rewiring
/// `mef.output` and erasing the staging op.
LogicalResult finalizeStagingOp(DeviceGraphStagingOp stagingOp, ModelOp model) {
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
    graphDevice = findGraphDevice(inputs);

  if (!graphDevice)
    return stagingOp.emitError("device-graph finalization: none of the staged "
                               "graph's inputs or outputs carries a device");

  DeviceContextValue deviceContext;
  for (auto [deviceRef, execDevice] :
       llvm::zip(model.getDevices(), model.getExecuteDevices())) {
    if (deviceRef == graphDevice) {
      deviceContext = cast<DeviceContextValue>(execDevice);
      break;
    }
  }
  if (!deviceContext)
    return stagingOp.emitError("device-graph finalization: no execute device "
                               "context matches the staged graph's device ")
           << graphDevice;

  IRRewriter rewriter(stagingOp);
  Location loc = stagingOp.getLoc();
  MLIRContext *ctx = rewriter.getContext();

  // The `mgp.device_graph.create` op is compiled like any other kernel by
  // wrapping it in a `mogg.generic.compiled.region`.
  auto graphType = DeviceGraphType::get(ctx, inputs, graphOutputs);

  SmallVector<Value> regionOperands{theContext, deviceContext};
  llvm::append_range(regionOperands, inputs);

  auto regionOpResult = wrapDeviceGraphBuildRegion(
      rewriter, loc, deviceContext.getType().getDevice(), TypeRange{graphType},
      regionOperands, "mgp.device_graph.create",
      stagingOp.getConstantResources(),
      [&](OpBuilder &bodyBuilder, Location bodyLoc,
          ValueRange unpacked) -> FailureOr<SmallVector<Value>> {
        Value contextUnpacked = unpacked[0];
        Value deviceContextUnpacked = unpacked[1];
        ValueRange inputsUnpacked = unpacked.drop_front(2);

        // The graph is cached in the state context, so an execution that hits
        // reuses the graph a previous one built instead of recording it again.
        Value cache =
            DeviceGraphCacheOp::create(bodyBuilder, bodyLoc, contextUnpacked);

        // Build the create op inside the region body, moving the staged body
        // into its build body. The create builder makes the body block with
        // [builder, context, inputs...] args; the callback splices the staged
        // ops in, rebinds the staged block args, and yields the graph outputs.
        auto createOp = DeviceGraphCreateOp::create(
            bodyBuilder, bodyLoc, contextUnpacked, deviceContextUnpacked, cache,
            inputsUnpacked,
            [&](OpBuilder &innerBuilder, Location innerLoc, Value graphBuilder,
                Value /*contextArg*/, ValueRange inputArgs) {
              Block *createBody = innerBuilder.getInsertionBlock();
              createBody->getOperations().splice(
                  createBody->end(), stagingBody.getOperations(),
                  stagingBody.begin(), std::prev(stagingBody.end()));

              // Yield before rebinding the staged block args below: a staged
              // body may yield one of its own inputs, and
              // `replaceAllUsesWith` only rewrites uses that already exist.
              DeviceGraphYieldOp::create(innerBuilder, innerLoc, graphOutputs);

              stagingBuilderArg.replaceAllUsesWith(graphBuilder);
              unsigned createArgIdx = 0;
              for (auto [stagingArg, stagingInput] :
                   llvm::zip_equal(stagingInputArgs, stagingOp.getInputs())) {
                if (isChain(stagingInput))
                  continue;
                stagingArg.replaceAllUsesWith(inputArgs[createArgIdx++]);
              }

              // We're transitioning to an SSACFG based operation, so we need to
              // order the operations in the body to enforce dominance
              // requirements.
              mlir::sortTopologically(createBody);
            });

        if (failed(resolveBuilderCasts(createOp.getBuildBody())))
          return mlir::failure();

        return SmallVector<Value>{createOp.getOutGraph()};
      });
  if (failed(regionOpResult))
    return mlir::failure();
  auto regionOp = *regionOpResult;

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
  return mlir::success();
}

/// Collect the values used inside a collective graph `region` that are defined
/// outside of it. Chain values (dead pass-throughs) and stable-address slots
/// are left to later work.
static LogicalResult collectRegionInputs(Region &region,
                                         SmallVectorImpl<Value> &inputs) {
  llvm::SetVector<Value> captured;
  mlir::getUsedValuesDefinedAbove(region, captured);

  inputs = nonChainValues(captured.getArrayRef());
  return mlir::success();
}

/// A region of a `mgp.device_graph.collective` after we have computed its
/// captures and the non-chain values it yields.
struct RegionInterface {
  Region *region;
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
};

/// Build a `RegionInterface` for each graph body in `collectiveOp`, computing
/// the values it captures from outside and the value outputs it contributes.
static LogicalResult
createRegionInterfaces(DeviceGraphCollectiveOp collectiveOp,
                       SmallVectorImpl<RegionInterface> &interfaces) {
  interfaces.reserve(collectiveOp.getNumRegions());

  for (auto graph : collectiveOp.graphs()) {
    RegionInterface iface;
    iface.region = graph.bodyBlock->getParent();
    if (failed(collectRegionInputs(*iface.region, iface.inputs)))
      return mlir::failure();

    auto yieldOp =
        cast<DeviceGraphCollectiveYieldOp>(graph.bodyBlock->getTerminator());
    for (Value out : yieldOp.getOperands())
      if (!isChain(out))
        iface.outputs.push_back(out);

    interfaces.push_back(std::move(iface));
  }
  return mlir::success();
}

/// Map each device referenced by `collectiveOp` to the corresponding
/// execute-block device context in `model`.
static FailureOr<SmallVector<DeviceContextValue>>
collectDeviceContexts(DeviceGraphCollectiveOp collectiveOp, ModelOp model) {
  SmallVector<DeviceContextValue> deviceContexts;
  for (auto deviceRef : collectiveOp.getDevices()) {
    DeviceContextValue deviceContext;
    for (auto [attr, execDevice] :
         llvm::zip(model.getDevices(), model.getExecuteDevices())) {
      if (attr == deviceRef) {
        deviceContext = cast<DeviceContextValue>(execDevice);
        break;
      }
    }
    if (!deviceContext)
      return collectiveOp.emitError(
                 "device-graph finalization: no execute device context matches "
                 "device ")
             << deviceRef;
    deviceContexts.push_back(deviceContext);
  }
  return deviceContexts;
}

/// Lower a single `mgp.device_graph.collective` op into a
/// `mgp.device_graph.collective.create` wrapped in a
/// `mogg.generic.compiled.region`, plus a bare
/// `mgp.device_graph.collective.execute` that replays it.
LogicalResult finalizeCollectiveOp(DeviceGraphCollectiveOp collectiveOp,
                                   ModelOp model) {
  Location loc = collectiveOp.getLoc();
  MLIRContext *ctx = collectiveOp->getContext();
  IRRewriter rewriter(collectiveOp);

  Value theContext = model.getExecuteContext();

  // Map each region to its execute-block device context.
  auto deviceContextsResult = collectDeviceContexts(collectiveOp, model);
  if (failed(deviceContextsResult))
    return mlir::failure();

  SmallVector<DeviceContextValue> deviceContexts =
      std::move(*deviceContextsResult);

  // Determine per-region inputs and outputs. Chain outputs are dropped from the
  // graph signature and later rewired to the execute op's fresh chain.
  SmallVector<RegionInterface> regionInterfaces;
  if (failed(createRegionInterfaces(collectiveOp, regionInterfaces)))
    return failure();

  // Flatten per-region inputs in region order for the new create/execute ops.
  SmallVector<Value> allInputs;
  SmallVector<unsigned> inputCounts;
  inputCounts.reserve(regionInterfaces.size());
  for (RegionInterface &info : regionInterfaces) {
    inputCounts.push_back(info.inputs.size());
    llvm::append_range(allInputs, info.inputs);
  }

  auto graphTypes =
      llvm::map_to_vector(regionInterfaces, [&](RegionInterface &info) -> Type {
        return DeviceGraphType::get(ctx, info.inputs, info.outputs);
      });

  // Wrap the build work in a compiled region: context + all device contexts +
  // all graph inputs enter via the async ABI, and the region yields the built
  // graphs (one per device).
  SmallVector<Value> regionOperands{theContext};
  llvm::append_range(regionOperands, deviceContexts);
  llvm::append_range(regionOperands, allInputs);

  auto regionOpResult = wrapDeviceGraphBuildRegion(
      rewriter, loc, deviceContexts.front().getType().getDevice(),
      TypeRange(graphTypes), regionOperands, collectiveOp.getOperationName(),
      collectiveOp.getConstantResources(),
      [&](OpBuilder &bodyBuilder, Location bodyLoc,
          ValueRange unpacked) -> FailureOr<SmallVector<Value>> {
        Value contextUnpacked = unpacked[0];
        SmallVector<Value> deviceContextsUnpacked =
            llvm::to_vector(unpacked.slice(1, deviceContexts.size()));
        SmallVector<Value> inputsUnpacked =
            llvm::to_vector(unpacked.drop_front(1 + deviceContexts.size()));

        Value cache =
            DeviceGraphCacheOp::create(bodyBuilder, bodyLoc, contextUnpacked);

        auto createOp = DeviceGraphCollectiveCreateOp::create(
            bodyBuilder, bodyLoc, contextUnpacked, deviceContextsUnpacked,
            cache, inputsUnpacked, inputCounts,
            [&](OpBuilder &innerBuilder, Location innerLoc, unsigned regionIdx,
                Value graphBuilder, Value /*contextArg*/,
                ValueRange inputArgs) {
              RegionInterface &info = regionInterfaces[regionIdx];
              Region &oldRegion = *info.region;
              Block &oldBlock = oldRegion.front();

              IRMapping mapper;
              mapper.map(oldRegion.getArgument(0), graphBuilder);
              mapper.map(info.inputs, inputArgs);

              for (Operation &op : oldBlock.without_terminator())
                innerBuilder.clone(op, mapper);

              auto oldYield = oldBlock.getTerminator();
              SmallVector<Value> newOutputs;
              newOutputs.reserve(info.outputs.size());
              for (Value out : oldYield->getOperands()) {
                if (isChain(out))
                  continue;
                Value mapped = mapper.lookupOrDefault(out);
                newOutputs.push_back(mapped);
              }

              DeviceGraphYieldOp::create(innerBuilder, innerLoc, newOutputs);
            });

        for (Region &createRegion : createOp.getGraphs())
          if (failed(resolveBuilderCasts(createRegion)))
            return mlir::failure();

        return llvm::to_vector_of<Value>(createOp.getOutGraphs());
      });

  if (failed(regionOpResult))
    return mlir::failure();
  auto regionOp = *regionOpResult;

  // The output types is the flattened sequence of output types for each
  // graph.
  SmallVector<Type> outputTypes;
  for (RegionInterface &info : regionInterfaces)
    llvm::append_range(outputTypes, TypeRange(info.outputs));

  rewriter.setInsertionPoint(collectiveOp);
  auto executeOp = DeviceGraphCollectiveExecuteOp::create(
      rewriter, loc, /*chain=*/MEF::ChainType::get(ctx),
      /*outputs=*/TypeRange(outputTypes),
      /*graphs=*/regionOp.getResults(),
      /*inputs=*/allInputs);

  // Rewire the collective op's results: chain results map to the execute chain,
  // value results map to the execute value outputs in region/yield order.
  Value executeChain = executeOp.getChain();
  unsigned outputIdx = 0;
  for (Value result : collectiveOp.getResults()) {
    if (isChain(result))
      result.replaceAllUsesWith(executeChain);
    else
      result.replaceAllUsesWith(executeOp.getOutputs()[outputIdx++]);
  }

  rewriter.eraseOp(collectiveOp);
  return mlir::success();
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
      if (failed(finalizeStagingOp(stagingOp, model)))
        return signalPassFailure();

    for (DeviceGraphCollectiveOp collectiveOp : llvm::to_vector(
             model.getExecuteBlock()->getOps<DeviceGraphCollectiveOp>()))
      if (failed(finalizeCollectiveOp(collectiveOp, model)))
        return signalPassFailure();
  }
};

} // namespace
} // namespace M::MGP
