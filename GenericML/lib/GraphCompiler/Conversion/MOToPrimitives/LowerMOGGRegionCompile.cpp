//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Converts MO ops to MGP and MEF ops.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/DeviceBufferizationRegistry.h"
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/Interfaces/LowerableToMGPInterface.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/MGPTransforms/Utils.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "Support/DeviceSpecs.h"

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_LOWERMOGGREGIONCOMPILES
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

//===----------------------------------------------------------------------===//
// MOGraphLowering implementation
//===----------------------------------------------------------------------===//

namespace M::MGP {

namespace {

struct LowerMOGGRegionCompilesPass
    : public MGP::impl::LowerMOGGRegionCompilesBase<
          LowerMOGGRegionCompilesPass> {
  using LowerMOGGRegionCompilesBase::LowerMOGGRegionCompilesBase;
  Value getOrCreateDeviceContextExtractOp(
      OpBuilder &builder, const Location &loc, const Value &inChain,
      const Value &inContext, const size_t runtimeSlot,
      const DeviceRefAttr device, Region *region,
      llvm::SmallDenseMap<
          std::tuple<Region *, Value>,
          llvm::SmallDenseMap<DeviceRefAttr, MGP::DeviceContextExtractOp>>
          &deviceContextExtractOpMap);
  void rewriteExecuteCompiledOp(OpBuilder &builder, const Value &context,
                                size_t runtimeSlot, size_t module_index,
                                size_t kernel_index,
                                MOGG::ExecuteCompiledOp &oldKernel,
                                Value loadChain, Value deviceContext,
                                std::optional<size_t> cuda_runtime_slot);
  void runOnOperation() override;
};

// This method will take a map that keeps track of all deviceContextExtractions
// and their region/inChain/device. If it cannot find one, it will create one
// and save it. This allows us to lazily create an extract only when we need it.
Value LowerMOGGRegionCompilesPass::getOrCreateDeviceContextExtractOp(
    OpBuilder &builder, const Location &loc, const Value &inChain,
    const Value &inContext, const size_t runtimeSlot,
    const DeviceRefAttr device, Region *region,
    llvm::SmallDenseMap<
        std::tuple<Region *, Value>,
        llvm::SmallDenseMap<DeviceRefAttr, MGP::DeviceContextExtractOp>>
        &deviceContextExtractOpMap) {
  if (!deviceContextExtractOpMap.contains(std::tuple(region, inChain))) {
    deviceContextExtractOpMap[std::tuple(region, inChain)] =
        llvm::SmallDenseMap<DeviceRefAttr, MGP::DeviceContextExtractOp>();
  }
  if (!deviceContextExtractOpMap[std::tuple(region, inChain)].contains(
          device)) {
    deviceContextExtractOpMap[std::tuple(region, inChain)][device] =
        builder.create<MGP::DeviceContextExtractOp>(
            loc,
            TypeRange{MGP::OpaqueType::get(
                builder.getContext(),
                StringAttr::get(builder.getContext(),
                                device.getLabel() +
                                    std::to_string(device.getId()) +
                                    "_context"),
                std::nullopt)},
            /*inChain=*/inChain,
            /*theContext*/ inContext,
            /*aDeviceRuntimeSlot=*/runtimeSlot,
            /*bDevice=*/device);
  }
  return deviceContextExtractOpMap[std::tuple(region, inChain)][device];
}

// Rewrite the MOGG::ExecuteCompiledOp into a MGP::ExecuteGenericOp
void LowerMOGGRegionCompilesPass::rewriteExecuteCompiledOp(
    OpBuilder &builder, const Value &context, size_t runtimeSlot,
    size_t module_index, size_t kernel_index,
    MOGG::ExecuteCompiledOp &oldKernel, Value loadChain, Value deviceContext,
    std::optional<size_t> cuda_runtime_slot) {
  auto ctx = oldKernel.getOperation()->getContext();
  SmallVector<Value> outputs;
  SmallVector<Type> resultTypes;
  for (auto [resTy, res] :
       llvm::zip(oldKernel.getResultTypes(), oldKernel.getResults())) {
    outputs.push_back(res);
    resultTypes.push_back(resTy);
  }

  auto device = oldKernel.getDeviceRef();
  if (!device)
    device = DeviceRefAttr::getDefault(ctx);

  ASSERT_STREAM(device.getLabel() == kCPULabel,
                << "Only CPU ExecuteCompiledOps can be lowered further");

  auto newExecuteOp = builder.create<MGP::ExecuteGenericOp>(
      oldKernel.getLoc(), MEF::ChainType::get(ctx), resultTypes, context,
      loadChain, deviceContext, oldKernel->getOperands(), runtimeSlot,
      module_index, kernel_index, oldKernel.getSummary(),
      oldKernel.getIsAsync(), oldKernel.getHasCuda(),
      cuda_runtime_slot.value_or(0));

  uint64_t idx = 1;
  for (auto result : outputs)
    result.replaceAllUsesWith(newExecuteOp.getResult(idx++));

  oldKernel.erase();
}

// Take all the compiled JIT Kernels and lower them
// -- Lower MOGG::LoadCompiledKernelOp to MGP::ModuleLoadOp
//    1. Determine the potential insertion point based off following rules
//       a. If there is ModuleLoadOp for the device --> After ModuleLoadOp +
//       keep track of Module Index for the host device (skipping non-host)
//       b. If no ModuleLoadOp for the specific device --> After
//       RuntimeCreateOp
//    2. Housekeeping to keep track of the kernel index
// -- Lower MOGG::ExecuteCompiledOp to MGP::ExecuteGenericOp
//    1. Special consideration has to be taken into account for init block and
//       ordering of the operations
//    2. Both init and execute blocks are processed.
void LowerMOGGRegionCompilesPass::runOnOperation() {

  MLIRContext *ctx = getOperation().getContext();
  for (auto model : getOperation().getOps<MGP::ModelOp>()) {
    Block &init = *model.getInitBlock();
    Block &execute = *model.getExecuteBlock();
    OpBuilder builder{ctx};

    // Setup the exec state
    ExecuteState execState(ctx, model);

    // MOGG::LoadCompiledKernelOp only on CPU and no other device. We just
    // need to make sure we use and count the right one for moduleIndex
    const auto device = DeviceRefAttr::get(ctx, kCPULabel, 0);
    const auto cudaDevice = DeviceRefAttr::get(ctx, kCUDALabel, 0);
    std::optional<size_t> cuda_runtime_slot;

    // Track the RuntimeCreateOp and ModuleLoadOp and their last use
    // Mainly used to make sure we target the right module load and device
    // runtime for the executions
    llvm::SmallDenseMap<DeviceRefAttr, MGP::RuntimeCreateOp>
        lastDeviceRuntimeOp;
    llvm::SmallDenseMap<DeviceRefAttr, MGP::ModuleLoadOp>
        lastDeviceModuleLoadOp;
    llvm::SmallDenseMap<DeviceRefAttr, uint64_t> deviceModuleIndex;
    for (auto op : init.getOps<MGP::RuntimeCreateOp>()) {
      lastDeviceRuntimeOp[op.getDevice()] = op;
      deviceModuleIndex[op.getDevice()] = 0;
    }

    // Stores for this model op, all the device context extractions.
    llvm::SmallDenseMap<
        std::tuple<Region *, Value>,
        llvm::SmallDenseMap<DeviceRefAttr, MGP::DeviceContextExtractOp>>
        deviceContextExtractOpMap;

    if (lastDeviceRuntimeOp.contains(cudaDevice))
      cuda_runtime_slot = lastDeviceRuntimeOp[cudaDevice].getSlot();
    for (auto op : init.getOps<MGP::ModuleLoadOp>()) {
      lastDeviceModuleLoadOp[op.getDevice()] = op;
      ASSERT_STREAM(deviceModuleIndex.contains(op.getDevice()),
                    << "Cannot have a ModuleLoadOp for a device: "
                    << op.getDevice()
                    << " without a corresponding RuntimeCreateOp");
      deviceModuleIndex[op.getDevice()]++;
    }
    llvm::SmallDenseMap<DeviceRefAttr, llvm::StringMap<size_t>>
        deviceKernelIndices;
    llvm::SmallDenseMap<DeviceRefAttr, SmallVector<StringAttr>>
        deviceKernelNameAttrs;
    MGP::ModuleLoadOp loadArchive;
    for (auto loadKernels : llvm::make_early_inc_range(
             init.getOps<MOGG::LoadCompiledKernelOp>())) {

      // Determine the kernelIndices and kernelNameAttrs
      deviceKernelIndices[device] = llvm::StringMap<size_t>();
      deviceKernelNameAttrs[device] = SmallVector<StringAttr>();
      for (StringAttr attr :
           loadKernels.getSymbols().getAsRange<StringAttr>()) {
        deviceKernelIndices[device][attr] = deviceKernelIndices[device].size();
        deviceKernelNameAttrs[device].push_back(attr);
      }
    }

    // Rewrite the MOGG::LoadCompiledKernelOp in init blocks.
    for (auto loadKernels : llvm::make_early_inc_range(
             init.getOps<MOGG::LoadCompiledKernelOp>())) {
      Operation *insertionOp = lastDeviceRuntimeOp[device].getOperation();
      size_t runtimeSlot = lastDeviceRuntimeOp[device].getSlot();
      Value opContext = lastDeviceRuntimeOp[device].getTheContext();
      Value inChain = lastDeviceRuntimeOp[device].getResult();
      if (lastDeviceModuleLoadOp.contains(device)) {
        insertionOp = lastDeviceModuleLoadOp[device].getOperation();
        runtimeSlot = lastDeviceModuleLoadOp[device].getSlot();
        opContext = lastDeviceModuleLoadOp[device].getTheContext();
        inChain = lastDeviceModuleLoadOp[device].getResult();
      }

      // Rewrite the MOGG::LoadCompiledKernelOp into a MGP::ModuleLoadOp
      builder.setInsertionPointAfter(insertionOp);
      loadArchive = builder.create<MGP::ModuleLoadOp>(
          insertionOp->getLoc(),
          /*inChain=*/inChain,
          /*theContext=*/opContext,
          /*aSlot=*/runtimeSlot,
          /*bArchive=*/loadKernels.getValue(),
          /*cKernelNames=*/
          MultiLineStringAttr::get(ctx, deviceKernelNameAttrs[device]),
          /*dConstantSymbols=*/
          MultiLineStringAttr::get(ctx, SmallVector<StringAttr>()),
          /*eDevice=*/device,
          /*fModuleIndex=*/deviceModuleIndex[device]);
      inChain.replaceAllUsesExcept(loadArchive.getResult(), loadArchive);
      loadKernels.erase();
    }

    // Rewrite the MOGG::ExecuteCompiledOp in both init and execute blocks.
    MEF::OutputOp mefOutputOp;
    for (auto op : init.getOps<MEF::OutputOp>())
      mefOutputOp = op;
    init.walk([&](MOGG::ExecuteCompiledOp kernel) {
      // Shunt all the executes for Init stage at the very end of init
      builder.setInsertionPoint(mefOutputOp);
      Value inChain;
      if (lastDeviceModuleLoadOp.find(device) != lastDeviceModuleLoadOp.end())
        inChain = lastDeviceModuleLoadOp[device].getOutChain();
      else
        inChain = lastDeviceRuntimeOp[device].getOutChain();
      Value inDeviceContext;
      if (kernel.getHasCuda()) {
        assert(lastDeviceRuntimeOp.contains(cudaDevice) &&
               "Must have a mgp.runtime.create which uses a cuda device");
        // lazily insert extractions so we don't need to pipe deviceContext
        // everywhere.
        // In the init, we have to be safe and wait on all operands of original
        // kernel for extraction.
        auto extractionChain = builder.create<GRT::ChainCreateOp>(
            kernel.getLoc(), kernel.getOperands());
        inDeviceContext = getOrCreateDeviceContextExtractOp(
            builder, mefOutputOp.getLoc(), extractionChain,
            lastDeviceRuntimeOp[cudaDevice].getTheContext(),
            lastDeviceRuntimeOp[cudaDevice].getSlot(), cudaDevice,
            kernel->getParentRegion(), deviceContextExtractOpMap);
      } else {
        auto extractionChain = builder.create<GRT::ChainCreateOp>(
            kernel.getLoc(), kernel.getOperands());
        inDeviceContext = getOrCreateDeviceContextExtractOp(
            builder, mefOutputOp.getLoc(), extractionChain,
            lastDeviceRuntimeOp[device].getTheContext(),
            lastDeviceRuntimeOp[device].getSlot(), device,
            kernel->getParentRegion(), deviceContextExtractOpMap);
      }
      rewriteExecuteCompiledOp(
          builder, lastDeviceRuntimeOp[device].getTheContext(),
          lastDeviceRuntimeOp[device].getSlot(), deviceModuleIndex[device],
          deviceKernelIndices[device][kernel.getKernelName()], kernel, inChain,
          inDeviceContext, cuda_runtime_slot);
    });
    Value executeContext = nullptr;
    Region *parentRegion = nullptr;
    execute.walk([&](MOGG::ExecuteCompiledOp kernel) {
      builder.setInsertionPoint(kernel);
      // For execute block, need to account for the bounding region in the
      // case of control flow or other bounding operations.
      if (parentRegion != kernel->getParentRegion()) {
        parentRegion = kernel->getParentRegion();
        executeContext = nullptr;
        for (Value arg : parentRegion->getArguments()) {
          if (isa<MGP::ContextType>(arg.getType()))
            executeContext = arg;
        }
        ASSERT_STREAM(executeContext,
                      << "Cannot find execute context in bounding region of "
                         "generic.execute");
      }
      auto inChain = builder.create<GRT::ChainCreateOp>(kernel.getLoc());
      Value inDeviceContext;
      if (kernel.getHasCuda()) {
        assert(lastDeviceRuntimeOp.contains(cudaDevice) &&
               "Must have a mgp.runtime.create which uses a cuda device");
        // lazily insert extractions so we don't need to pipe deviceContext
        // everywhere.
        inDeviceContext = getOrCreateDeviceContextExtractOp(
            builder, kernel.getLoc(), inChain, executeContext,
            lastDeviceRuntimeOp[cudaDevice].getSlot(), cudaDevice,
            kernel->getParentRegion(), deviceContextExtractOpMap);
      } else {
        inDeviceContext = getOrCreateDeviceContextExtractOp(
            builder, kernel.getLoc(), inChain, executeContext,
            lastDeviceRuntimeOp[device].getSlot(), device,
            kernel->getParentRegion(), deviceContextExtractOpMap);
      }
      rewriteExecuteCompiledOp(
          builder, executeContext, lastDeviceRuntimeOp[device].getSlot(),
          deviceModuleIndex[device],
          deviceKernelIndices[device][kernel.getKernelName()], kernel, inChain,
          inDeviceContext, cuda_runtime_slot);
    });
  }
}

} // namespace

} // namespace M::MGP
