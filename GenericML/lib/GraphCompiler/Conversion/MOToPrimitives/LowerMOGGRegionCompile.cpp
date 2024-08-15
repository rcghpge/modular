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
#include "GenericML/MGPTransforms/FixupIsolatedFromAbove.h"
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

/// DeviceContextState is used to track device contexts within a model op
struct DeviceContextState {
  // State information to keep track of a deviceContext and the region it exists
  // in
  llvm::SmallDenseMap<std::tuple<Region *, DeviceRefAttr>, Value>
      regionDeviceToContextMap;
  llvm::SmallDenseMap<DeviceRefAttr, Value> initInputContextMap;

  // Initializes the starting value
  void init(MGP::ModelOp model) {
    // Populate arguments for init region
    for (const auto &[device, context] :
         llvm::zip(model.getDevices(), model.getInitDevices()))
      initInputContextMap[dyn_cast<DeviceRefAttr>(device)] = context;

    // Populate starting value for init region using the output device-context
    // because the device_context can be created by an init-phase operation
    for (const auto &[device, context] :
         llvm::zip(model.getDevices(), model.getInitOutputDevices())) {
      Region *initRegion = model.getInitBlock()->getParent();
      DeviceRefAttr deviceAttr = dyn_cast<DeviceRefAttr>(device);
      setDeviceContext(deviceAttr, initRegion, context);
    }

    // Populate starting value for exec region using the input argument
    for (const auto &[device, context] :
         llvm::zip(model.getDevices(), model.getExecuteDevices())) {
      Region *execRegion = model.getExecuteBlock()->getParent();
      DeviceRefAttr deviceAttr = dyn_cast<DeviceRefAttr>(device);
      setDeviceContext(deviceAttr, execRegion, context);
    }
  }

  /// This function gets the DeviceContext defined for this region.
  Value getDeviceContext(DeviceRefAttr deviceAttr, Region *region) {
    assert(regionDeviceToContextMap.contains({region, deviceAttr}) &&
           "Region must contain the context as args");
    return regionDeviceToContextMap[{region, deviceAttr}];
  }

  /// This function sets the DeviceContext defined for this region.
  void setDeviceContext(DeviceRefAttr deviceAttr, Region *region, Value value) {
    regionDeviceToContextMap[{region, deviceAttr}] = value;
  }

  /// This function gets the DeviceContext from Init Args
  Value getDeviceContextFromInitArgs(DeviceRefAttr deviceAttr) {
    assert(initInputContextMap.contains(deviceAttr) &&
           "Args must contain the context dor the device as arg");
    return initInputContextMap[deviceAttr];
  }
};

struct LowerMOGGRegionCompilesPass
    : public MGP::impl::LowerMOGGRegionCompilesBase<
          LowerMOGGRegionCompilesPass> {
  using LowerMOGGRegionCompilesBase::LowerMOGGRegionCompilesBase;
  void rewriteExecuteCompiledOp(OpBuilder &builder, const Value &context,
                                size_t runtimeSlot, size_t module_index,
                                size_t kernel_index,
                                MOGG::ExecuteCompiledOp &oldKernel,
                                Value loadChain, Value deviceContext,
                                std::optional<size_t> cuda_runtime_slot);
  void runOnOperation() override;
  DeviceContextState deviceContexts;
};

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
  for (auto model :
       llvm::make_early_inc_range(getOperation().getOps<MGP::ModelOp>())) {
    Block &init = *model.getInitBlock();
    Block &execute = *model.getExecuteBlock();
    OpBuilder builder{ctx};

    // Setup the exec state
    ExecuteState execState(ctx, model);

    // MOGG::LoadCompiledKernelOp only on CPU and no other device. We just
    // need to make sure we use and count the right one for moduleIndex
    const auto hostDevice = DeviceRefAttr::get(ctx, kCPULabel, 0);

    // TODO: (GRA-831) Remove cudaDevice and cuda_runtime_slot from
    // generic.execute once device_context plumbed
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
    deviceContexts.init(model);

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
      deviceKernelIndices[hostDevice] = llvm::StringMap<size_t>();
      deviceKernelNameAttrs[hostDevice] = SmallVector<StringAttr>();
      for (StringAttr attr :
           loadKernels.getSymbols().getAsRange<StringAttr>()) {
        deviceKernelIndices[hostDevice][attr] =
            deviceKernelIndices[hostDevice].size();
        deviceKernelNameAttrs[hostDevice].push_back(attr);
      }
    }

    // Rewrite the MOGG::LoadCompiledKernelOp in init blocks.
    for (auto loadKernels : llvm::make_early_inc_range(
             init.getOps<MOGG::LoadCompiledKernelOp>())) {
      Operation *insertionOp = lastDeviceRuntimeOp[hostDevice].getOperation();
      size_t runtimeSlot = lastDeviceRuntimeOp[hostDevice].getSlot();
      Value opContext = lastDeviceRuntimeOp[hostDevice].getTheContext();
      Value inChain = lastDeviceRuntimeOp[hostDevice].getResult();
      if (lastDeviceModuleLoadOp.contains(hostDevice)) {
        insertionOp = lastDeviceModuleLoadOp[hostDevice].getOperation();
        runtimeSlot = lastDeviceModuleLoadOp[hostDevice].getSlot();
        opContext = lastDeviceModuleLoadOp[hostDevice].getTheContext();
        inChain = lastDeviceModuleLoadOp[hostDevice].getResult();
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
          MultiLineStringAttr::get(ctx, deviceKernelNameAttrs[hostDevice]),
          /*dConstantSymbols=*/
          MultiLineStringAttr::get(ctx, SmallVector<StringAttr>()),
          /*eDevice=*/hostDevice,
          /*fModuleIndex=*/deviceModuleIndex[hostDevice],
          /*gDebuginfo=*/loadKernels.getDebuginfo());
      inChain.replaceAllUsesExcept(loadArchive.getResult(), loadArchive);
      loadKernels.erase();
    }
    // Rewrite the MOGG::ExecuteCompiledOp in both init and execute blocks.
    // Shunt all the executes for Init stage at the very end of init
    builder.setInsertionPoint(init.getTerminator());
    init.walk([&](MOGG::ExecuteCompiledOp kernel) {
      Value inChain;
      if (lastDeviceModuleLoadOp.find(hostDevice) !=
          lastDeviceModuleLoadOp.end())
        inChain = lastDeviceModuleLoadOp[hostDevice].getOutChain();
      else
        inChain = lastDeviceRuntimeOp[hostDevice].getOutChain();
      auto device = kernel.getHasCuda() ? cudaDevice : hostDevice;
      Value inDeviceContext =
          deviceContexts.getDeviceContext(device, init.getParent());
      bool isCreator = inDeviceContext.getDefiningOp() == kernel;
      if (isCreator)
        inDeviceContext = deviceContexts.getDeviceContextFromInitArgs(device);
      rewriteExecuteCompiledOp(
          builder, lastDeviceRuntimeOp[hostDevice].getTheContext(),
          lastDeviceRuntimeOp[hostDevice].getSlot(),
          deviceModuleIndex[hostDevice],
          deviceKernelIndices[hostDevice][kernel.getKernelName()], kernel,
          inChain, inDeviceContext, cuda_runtime_slot);
      // Since this execute op created the deviceContext, we should update the
      // value mapping.
      if (isCreator)
        deviceContexts.init(model);
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
      Value inDeviceContext = deviceContexts.getDeviceContext(
          kernel.getHasCuda() ? cudaDevice : hostDevice, execute.getParent());
      rewriteExecuteCompiledOp(
          builder, executeContext, lastDeviceRuntimeOp[hostDevice].getSlot(),
          deviceModuleIndex[hostDevice],
          deviceKernelIndices[hostDevice][kernel.getKernelName()], kernel,
          inChain, inDeviceContext, cuda_runtime_slot);
    });
    // The introduction of generic execute may introduce unbound free-values.
    // Use fixupIsolatedFromAbove to rebind these.
    (void)fixupIsolatedFromAbove(builder, model);
    model.erase();
  }
}

} // namespace

} // namespace M::MGP
