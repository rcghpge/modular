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
#include "mlir/Analysis/TopologicalSortUtils.h"
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

/// Each stable create op's slot index.
using StableInfos = llvm::MapVector<DeviceGraphStableCreateOp, unsigned>;

/// Stable create ops and their assigned stable addresses.
struct StableAddrState {
  StableAddrState(const StableInfos &infos,
                  ArrayRef<BlockArgument> stableAddrs) {
    for (auto &[op, index] : infos)
      assignment[op] = stableAddrs[index];
  }

  Value addr(DeviceGraphStableCreateOp create) { return assignment.at(create); }

  /// Returns the stable get/place result for `operand` if it is a stable
  /// value.
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

    // On the source device, reuse the place's view instead of a stable.get.
    stableToGet[stable] = placed;
  }

private:
  llvm::SmallDenseMap<DeviceGraphStableCreateOp, Value> assignment;

  llvm::SmallDenseMap<DeviceGraphStableCreateOp, Value> stableToGet;
};

M::DeviceRefAttr getDevice(GenericCompiledRegionOp op) {
  return cast<MOHasDeviceInterface>(op.getOperation()).getExecutionDevice();
}

/// Collects the compiled regions and stable creates, giving each stable slot a
/// unique index.
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
    // CPU regions are host-side bookkeeping (e.g. wrapping pointers into
    // !mgp.tensor values), duplicated into each accelerator region.
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

/// Partitions `mgp.device_graph.staging` into a `mgp.device_graph.collective`
/// with one region per device.
LogicalResult partitionStagingOp(DeviceGraphStagingOp stagingOp,
                                 IRRewriter &rewriter) {
  Location loc = stagingOp.getLoc();

  // The collective is not a graph region, so toposort first.
  mlir::sortTopologically(&stagingOp.getBodyBlock());

  // The body is isolated, so enclosing values are read through its block args.
  IRMapping baseMapper;
  baseMapper.map(stagingOp.getBodyInputs(), stagingOp.getInputs());

  StableInfos stableIndices;
  SmallVector<GenericCompiledRegionOp> compiledRegions;

  if (failed(collectOperations(stagingOp, stableIndices, compiledRegions)))
    return failure();

  SmallVector<M::DeviceRefAttr> deviceOrder = collectDevices(compiledRegions);
  if (deviceOrder.empty())
    return stagingOp.emitOpError(
        "staging body has no compiled regions; nothing to partition");

  SmallVector<StableAddressType> stableSlotTypes = llvm::map_to_vector(
      stableIndices.keys(),
      [&](DeviceGraphStableCreateOp stable) { return stable.getStableType(); });

  // Staging yield operand -> result index in the collective.
  llvm::SmallDenseMap<Value, unsigned> resultMapping;
  unsigned resultCounter = 0;

  LogicalResult bodyStatus = success();

  // Each per-device callback replays that device's staged ops, adding
  // stable.get / stable.place as needed.
  rewriter.setInsertionPoint(stagingOp);
  auto collective = DeviceGraphCollectiveOp::create(
      rewriter, loc, deviceOrder, stableSlotTypes,
      [&](OpBuilder &builder, M::DeviceRefAttr device, BlockArgument builderArg,
          ArrayRef<BlockArgument> stableAddrs) {
        if (failed(bodyStatus))
          return;

        IRMapping mapper = baseMapper;

        StableAddrState stableState(stableIndices, stableAddrs);

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

          for (Value operand : compiledRegion.getOperands()) {
            if (auto resolved = resolveOperand(operand))
              mapper.map(operand, resolved);
          }

          auto cloned = cast<GenericCompiledRegionOp>(
              builder.clone(*compiledRegion, mapper));
          mapper.map(compiledRegion.getResults(), cloned.getResults());
        }

        // CPU work is in every region, so its yields are available everywhere;
        // device work yields only from its own region.
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
