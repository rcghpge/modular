//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

using namespace M::MGP;
using namespace M::MO;
using namespace M::MOGG;
using namespace mlir;

namespace {

/// Association of each stable create operation with its relevant info.
using StableInfos = llvm::MapVector<DeviceGraphStableCreateOp, unsigned>;

/// Association of stable create operations to the stable addresses assigned
/// to those operations.
struct StableAddrState {
  StableAddrState(const StableInfos &infos,
                  ArrayRef<BlockArgument> stableAddrs) {
    for (auto &[op, index] : infos)
      assignment[op] = stableAddrs[index];
  }

  /// Get the backing stable address.
  Value addr(DeviceGraphStableCreateOp create) { return assignment.at(create); }

  /// Find the associated DeviceGraphStableGetOp result for an operand, if that
  /// operand was a stable value produced by DeviceGraphStableCreateOp.
  Value get(OpBuilder &builder, Location loc, Value builderArg, Value operand) {
    auto stable = operand.getDefiningOp<DeviceGraphStableCreateOp>();
    if (!stable)
      return {};

    Value &get = stableToGet[stable];
    if (!get)
      get = DeviceGraphStableGetOp::create(builder, loc, addr(stable),
                                           builderArg);

    return get;
  }

  void place(OpBuilder &builder, Location loc, Value builderArg,
             DeviceGraphStableCreateOp stable, Value source) {
    Value address = addr(stable);
    Value placed = DeviceGraphStablePlaceOp::create(builder, loc, address,
                                                    source, builderArg);

    // On the source device the stable value is already published by
    // the place; reuse that view instead of emitting a redundant
    // stable.get.
    stableToGet[stable] = placed;
  }

private:
  /// Association of each DeviceGraphStableCreateOp to the stable address
  /// within the body region.
  llvm::SmallDenseMap<DeviceGraphStableCreateOp, Value> assignment;

  /// Mapping from a stable create op to the get/place operation producing
  /// its result.
  llvm::SmallDenseMap<DeviceGraphStableCreateOp, Value> stableToGet;
};

M::DeviceRefAttr getDevice(GenericCompiledRegionOp op) {
  return cast<MOHasDeviceInterface>(op.getOperation()).getExecutionDevice();
}

/// Collect all the compiled regions and stable addresses, assigning each stable
/// slot a unique index.
LogicalResult
collectOperations(DeviceGraphStagingOp staging, StableInfos &stables,
                  SmallVectorImpl<GenericCompiledRegionOp> &compiledRegions) {
  Block &body = staging.getBodyBlock();

  for (Operation &op : body.without_terminator()) {
    if (auto stable = dyn_cast<DeviceGraphStableCreateOp>(&op)) {
      stables.try_emplace(stable, static_cast<unsigned>(stables.size()));
    } else if (auto region = dyn_cast<GenericCompiledRegionOp>(&op)) {
      compiledRegions.push_back(region);
    } else {
      // Unknown ops inside a staging body are unexpected after staging.
      return op.emitOpError(
          "unexpected operation inside device-graph staging body");
    }
  }

  return success();
}

SmallVector<M::DeviceRefAttr>
collectDevices(ArrayRef<GenericCompiledRegionOp> compiledRegions) {
  llvm::SmallSetVector<M::DeviceRefAttr, 8> deviceOrder;

  for (auto region : compiledRegions) {
    // CPU compiled regions are host-side bookkeeping (e.g. wrapping pointers
    // into !mgp.tensor values), not device-graph work. They are duplicated
    // into each accelerator region instead of getting their own region.
    if (M::MO::isOnHost(region))
      continue;

    deviceOrder.insert(getDevice(region));
  }

  SmallVector<M::DeviceRefAttr> devices = deviceOrder.takeVector();
  llvm::sort(devices, [](M::DeviceRefAttr a, M::DeviceRefAttr b) {
    return std::tuple(a.getLabel(), a.getId()) <
           std::tuple(b.getLabel(), b.getId());
  });

  return devices;
}

/// Partition the single `mgp.device_graph.staging` op inside `model`'s execute
/// block into a `mgp.device_graph.collective` op with one region per device.
LogicalResult partitionStagingOp(DeviceGraphStagingOp stagingOp,
                                 IRRewriter &rewriter) {
  Location loc = stagingOp.getLoc();

  // Map staging body input block args back to the outer operands.  The body is
  // isolated, so every read of an enclosing value goes through these arguments.
  IRMapping baseMapper;
  baseMapper.map(stagingOp.getBodyInputs(), stagingOp.getInputs());

  // Walk the staging body and collect the work we need to partition.
  StableInfos stableIndices;
  SmallVector<GenericCompiledRegionOp> compiledRegions;

  if (failed(collectOperations(stagingOp, stableIndices, compiledRegions)))
    return failure();

  // Determine the ordered set of participating devices from the compiled
  // regions.  Sorting gives a stable region order.
  SmallVector<M::DeviceRefAttr> deviceOrder = collectDevices(compiledRegions);
  if (deviceOrder.empty())
    return stagingOp.emitOpError(
        "staging body has no compiled regions; nothing to partition");

  // Extract the slot types for the stable slot allocations.
  SmallVector<StableAddressType> stableSlotTypes = llvm::map_to_vector(
      stableIndices.keys(),
      [&](DeviceGraphStableCreateOp stable) { return stable.getStableType(); });

  // Mapping of the staging operation's yield operands to their result index in
  // the new DeviceGraphCollectiveOp.
  llvm::SmallDenseMap<Value, unsigned> resultMapping;
  unsigned resultCounter = 0;

  LogicalResult bodyStatus = success();

  // Build the collective with a per-device body callback.  The callback replays
  // the staged ops for its device, emits stable.get / stable.place as needed,
  // and produces the region terminator.
  rewriter.setInsertionPoint(stagingOp);
  auto collective = DeviceGraphCollectiveOp::create(
      rewriter, loc, deviceOrder, stableSlotTypes,
      [&](OpBuilder &builder, M::DeviceRefAttr device, BlockArgument builderArg,
          ArrayRef<BlockArgument> stableAddrs) {
        if (failed(bodyStatus))
          return;

        IRMapping mapper = baseMapper;

        StableAddrState stableState(stableIndices, stableAddrs);

        // Resolve a stable operand back to the stable get/place operation
        // that produced it.
        auto resolveOperand = [&](Value operand) -> Value {
          return stableState.get(builder, loc, builderArg, operand);
        };

        for (Operation &op : stagingOp.getBodyBlock().without_terminator()) {
          if (auto stable = dyn_cast<DeviceGraphStableCreateOp>(&op)) {
            if (stable.getDevice() != device)
              continue;

            Value source = mapper.lookupOrNull(stable.getTensor());
            if (!source) {
              bodyStatus = stable.emitOpError(
                  "stable.create source was not materialized on its device");
              return;
            }

            stableState.place(builder, loc, builderArg, stable, source);
            continue;
          }

          auto compiledRegion = dyn_cast<GenericCompiledRegionOp>(&op);
          if (!compiledRegion) {
            bodyStatus = op.emitOpError(
                "unexpected operation inside device-graph staging body");
            return;
          }

          auto opDev = getDevice(compiledRegion);
          bool isHostOp = M::MO::isOnHost(compiledRegion);

          // Host work is duplicated into every accelerator region; device
          // work is placed only on its own region.
          if (opDev != device && !isHostOp)
            continue;

          // Materialize stable.gets for any stable values consumed by this
          // compiled region on this device.
          for (Value operand : compiledRegion.getOperands()) {
            if (auto resolved = resolveOperand(operand))
              mapper.map(operand, resolved);
          }

          auto cloned = cast<GenericCompiledRegionOp>(
              builder.clone(*compiledRegion, mapper));
          mapper.map(compiledRegion.getResults(), cloned.getResults());
        }

        // Determine this region's terminator operands from the original
        // staging yield. CPU work is cloned into every accelerator region, so
        // its yields are available everywhere; device work is only yielded from
        // its own region.
        SmallVector<Value> regionYields;
        for (Value origYield : stagingOp.getYieldOp().getOperands()) {
          auto producer = origYield.getDefiningOp<GenericCompiledRegionOp>();
          if (!producer) {
            bodyStatus = stagingOp.getYieldOp().emitOpError(
                "staging yield value is not produced by a compiled region");
            return;
          }

          Value cloned = mapper.lookupOrNull(origYield);
          if (!cloned)
            continue;

          regionYields.push_back(cloned);
          {
            auto emplaced = resultMapping.try_emplace(origYield, resultCounter);
            if (emplaced.second)
              ++resultCounter;
          }
        }

        DeviceGraphCollectiveYieldOp::create(builder, loc, regionYields);
      });

  if (failed(bodyStatus)) {
    rewriter.eraseOp(collective);
    return failure();
  }

  for (auto [yielded, result] : llvm::zip(stagingOp.getYieldOp().getOperands(),
                                          stagingOp.getResults())) {
    auto collectiveResult = collective.getResult(resultMapping.at(yielded));
    rewriter.replaceAllUsesWith(result, collectiveResult);
  }

  rewriter.eraseOp(stagingOp);
  return success();
}

} // namespace

namespace M::MGP {

#define GEN_PASS_DEF_PARTITIONDEVICEGRAPHCOLLECTIVEPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

class PartitionDeviceGraphCollectivePass
    : public impl::PartitionDeviceGraphCollectivePassBase<
          PartitionDeviceGraphCollectivePass> {
  using Base::Base;

  void runOnOperation() override {
    ModelOp model = getOperation();

    if (!model.getIsDeviceGraph())
      return;

    IRRewriter rewriter(&getContext());

    auto stagingOps = llvm::to_vector(
        model.getExecuteBlock()->getOps<DeviceGraphStagingOp>());

    for (DeviceGraphStagingOp staging : stagingOps)
      if (failed(partitionStagingOp(staging, rewriter)))
        signalPassFailure();
  }
};

} // namespace
} // namespace M::MGP
