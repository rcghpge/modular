//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

using namespace M::MGP;
using namespace M::MOGG;
using namespace mlir;

namespace {

/// Reconcile `v` to `target` by inserting a single `mogg.async` pack/unpack at
/// the dissolved region boundary. A compiled region wraps its operands into
/// `!mogg.async<T>` on entry and packs its results back out, so when the
/// boundary is dissolved the verbatim body's pack/unpack ops must still see an
/// `!mogg.async<...>` while the outer values are plain `T` (and vice-versa for
/// results). Returns nullptr (after emitting an error) if the types are not
/// bridgeable by one async wrap/unwrap.
Value reconcileAcrossBoundary(OpBuilder &builder, Location loc, Value value,
                              Type target, Operation *errorOp) {
  Type valueType = value.getType();
  if (valueType == target)
    return value;

  // `target` is `!mogg.async<vTy>`: pack `v` up so the body's unpack still has
  // an async input.
  if (auto targetAsync = dyn_cast<AsyncType>(target);
      targetAsync && targetAsync.getElementType() == valueType)
    return AsyncPackOp::create(builder, loc, target, value);

  // `v` is `!mogg.async<target>`: unpack it down for the outside consumers.
  if (auto vAsync = dyn_cast<AsyncType>(valueType);
      vAsync && vAsync.getElementType() == target)
    return AsyncUnpackOp::create(builder, loc, target, value);

  errorOp->emitError("device-graph dependency conversion: cannot reconcile ")
      << valueType << " with " << target
      << " across the region boundary (expected an async wrap/unwrap)";

  return nullptr;
}

/// Populate the body of a `mgp.device_graph.dependencies` op (via its callback
/// builder, so `builder` is already positioned in the empty body block, which
/// already carries the `graphBuilder` block arg) from the verbatim contents of
/// `region`, dissolving the compiled region's dataflow boundary: the region's
/// block args are remapped to the actual outer values (`oldValueToNew`
/// redirects values produced by already-converted sibling regions to their new
/// `bodyResults`), and `mogg.async` pack/unpack ops are inserted at the seam so
/// the body still type-checks. Device-context block args are special-cased:
/// they are re-sourced from `graphBuilder` via a
/// `builtin.unrealized_conversion_cast` so the body is tied to the device graph
/// under construction rather than the outer device context. Emits the
/// `dependencies.yield`. Returns failure if a boundary type cannot be
/// reconciled.
LogicalResult buildDependenciesBody(OpBuilder &builder, Location loc,
                                    Value graphBuilder,
                                    GenericCompiledRegionOp region,
                                    const IRMapping &oldValueToNew) {
  Block &srcBlock = region.getBody().front();

  // The builder -> device context bridges, created lazily (one per context
  // type) so context-free bodies emit no cast. They are unrealized for now and
  // resolved later, when the recording stub is rewritten to accept a
  // `!mgp.device_graph.builder` directly.
  llvm::SmallDenseMap<Type, Value> deviceContextCast;

  // Entry mediation: map each region block arg to the (reconciled) outer value
  // it was passed. Values produced by already-converted siblings are redirected
  // to their new `bodyResults`; everything else (model inputs, outer SSA) is
  // referenced directly since this op is not `IsolatedFromAbove`.
  IRMapping mapper;
  for (auto [blockArg, operand] :
       llvm::zip(srcBlock.getArguments(), region.getOperands())) {
    // A device-context block arg (the launch context) -- which the region's
    // ABI wraps as `!mogg.async<!mgp.device_context>` -- is re-sourced from the
    // builder block arg rather than the outer device context, so the body is
    // self-contained. The cast is reconciled across the boundary exactly like
    // any other value (re-packed to async here, the body's verbatim
    // `mogg.async.unpack` then folds the round-trip away in Stage 3).
    Type elementType = blockArg.getType();
    if (auto async = llvm::dyn_cast<AsyncType>(elementType))
      elementType = async.getElementType();

    auto ctxType = dyn_cast<DeviceContextType>(elementType);
    if (ctxType && ctxType.getDevice().getLabel() != M::kCPULabel) {
      Value &cast = deviceContextCast[elementType];
      if (!cast)
        cast = UnrealizedConversionCastOp::create(builder, loc, elementType,
                                                  graphBuilder)
                   .getResult(0);
      Value mapped = reconcileAcrossBoundary(builder, loc, cast,
                                             blockArg.getType(), region);
      if (!mapped)
        return failure();
      mapper.map(blockArg, mapped);
      continue;
    }
    Value src = oldValueToNew.lookupOrDefault(operand);
    Value mapped =
        reconcileAcrossBoundary(builder, loc, src, blockArg.getType(), region);
    if (!mapped)
      return failure();
    mapper.map(blockArg, mapped);
  }

  // Clone the body verbatim (clone remaps operand uses through `mapper`).
  for (Operation &op : srcBlock.without_terminator())
    builder.clone(op, mapper);

  // Exit mediation: the `mogg.output` operands are the region's (async-packed)
  // results; reconcile each down to the plain region result type so the yield
  // matches the op's `bodyResults`.
  OutputOp outputOp = region.getBodyTerminator();
  SmallVector<Value> yielded;
  yielded.reserve(outputOp.getNumOperands());
  for (auto [operand, resultType] :
       llvm::zip(outputOp.getOperands(), region.getResultTypes())) {
    Value mapped = reconcileAcrossBoundary(
        builder, loc, mapper.lookupOrDefault(operand), resultType, region);
    if (!mapped)
      return failure();
    yielded.push_back(mapped);
  }
  DeviceGraphDependenciesYieldOp::create(builder, loc, yielded);
  return success();
}

/// Replace every `mogg.generic.compiled.region` in each device region of
/// `collectiveOp` with a `mgp.device_graph.dependencies` op, preserving the
/// per-device ordering and dissolving each compiled region's dataflow boundary.
LogicalResult convertCollectiveRegions(DeviceGraphCollectiveOp collectiveOp) {
  llvm::MapVector<GenericCompiledRegionOp, DeviceGraphDependenciesOp> oldToNew;
  IRMapping oldValueToNew;

  // Each device region has its own `!mgp.device_graph.builder` block argument.
  // Convert the compiled regions inside each region independently; edges across
  // regions are derived below using the same global `oldToNew` map.
  for (DeviceGraphCollectiveOp::CollectiveRegion graph :
       collectiveOp.graphs()) {
    Value regionBuilder = graph.builder;

    for (auto region : graph.getOps<GenericCompiledRegionOp>()) {
      OpBuilder builder(region);

      LogicalResult bodyResult = success();
      auto depsOp = DeviceGraphDependenciesOp::create(
          builder, region.getLoc(), /*graphBuilder=*/regionBuilder,
          /*dependencies=*/ValueRange{},
          [&](OpBuilder &bodyBuilder, Location loc, Value graphBuilder) {
            bodyResult = buildDependenciesBody(bodyBuilder, loc, graphBuilder,
                                               region, oldValueToNew);
          });

      if (failed(bodyResult))
        return failure();

      oldToNew[region] = depsOp;
      oldValueToNew.map(region.getResults(), depsOp.getBodyResults());
    }
  }

  // Derive dependency edges. Operands defined by another converted region
  // (within the same or another device region) become edges to that region's
  // node; other operands are model inputs or outer SSA and contribute no edge.
  for (auto [region, dependencies] : oldToNew) {
    llvm::SmallSetVector<Value, 8> edges;
    for (Value operand : region.getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (auto producer = dyn_cast_if_present<GenericCompiledRegionOp>(def))
        if (auto it = oldToNew.find(producer); it != oldToNew.end())
          edges.insert(it->second.getNode());
    }
    dependencies.getDependenciesMutable().assign(edges.getArrayRef());
  }

  // Rewire all uses of the original regions' results to the new body results,
  // then erase the originals.
  for (auto [region, dependencies] : oldToNew) {
    for (auto [oldResult, newResult] :
         llvm::zip(region.getResults(), dependencies.getBodyResults()))
      oldResult.replaceAllUsesWith(newResult);
  }

  SetVector<Attribute> constantResources;
  for (auto region : oldToNew.keys()) {
    if (auto resources = region.getConstantResources())
      constantResources.insert_range(resources);
    region.erase();
  }

  Builder builder(collectiveOp->getContext());
  auto newAttr = builder.getArrayAttr(constantResources.getArrayRef());
  collectiveOp.setConstantResourcesAttr(newAttr);

  return success();
}

/// Convert the `mogg.generic.compiled.region` ops staged by
/// `mgp-stage-device-graphs` (or already bundled by
/// `mgp.device_graph.collective`) into `mgp.device_graph.dependencies` ops,
/// capturing the dependency DAG explicitly.
LogicalResult computeDeviceGraphDependencies(ModelOp model) {
  Block *execute = model.getExecuteBlock();

  // The same lowering applies to `mgp.device_graph.collective`, where each
  // device region is processed separately.
  for (DeviceGraphCollectiveOp collectiveOp :
       llvm::make_early_inc_range(execute->getOps<DeviceGraphCollectiveOp>())) {
    if (failed(convertCollectiveRegions(collectiveOp)))
      return failure();
  }
  return success();
}

} // namespace

namespace M::MGP {

#define GEN_PASS_DEF_COMPUTEDEVICEGRAPHDEPENDENCIESPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

class ComputeDeviceGraphDependenciesPass
    : public impl::ComputeDeviceGraphDependenciesPassBase<
          ComputeDeviceGraphDependenciesPass> {
  using Base::Base;

  void runOnOperation() override {
    ModelOp theModel = getOperation();

    // Device-graph synthesis; runs after `mgp-stage-device-graphs`.
    if (theModel.getIsDeviceGraph())
      if (failed(computeDeviceGraphDependencies(theModel)))
        return signalPassFailure();
  }
};

} // namespace
} // namespace M::MGP
