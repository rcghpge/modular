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
           "Args must contain the context or the device as arg");
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
                                Value deviceContext);
  void runOnOperation() override;
  DeviceContextState deviceContexts;
};

// Rewrite the MOGG::ExecuteCompiledOp into a MGP::ExecuteGenericOp
void LowerMOGGRegionCompilesPass::rewriteExecuteCompiledOp(
    OpBuilder &builder, const Value &context, size_t runtimeSlot,
    size_t module_index, size_t kernel_index,
    MOGG::ExecuteCompiledOp &oldKernel, Value deviceContext) {
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

  auto newExecuteOp = builder.create<MGP::ExecuteGenericOp>(
      oldKernel.getLoc(), resultTypes, context, deviceContext,
      oldKernel->getOperands(), runtimeSlot, module_index, kernel_index,
      oldKernel.getSummary(), oldKernel.getIsAsync());

  for (auto [idx, result] : llvm::enumerate(outputs))
    result.replaceAllUsesWith(newExecuteOp.getResult(idx));

  oldKernel.erase();
}

// Take all the compiled JIT Kernels and lower them
// -- Lower MOGG::LoadCompiledKernelOp to MGP::ModuleLoadOp
//    1. Determine the potential insertion point based off following rules
//       --> After RuntimeCreateOp
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

    // Set up the exec state
    ExecuteState execState(ctx, model);

    // MOGG::LoadCompiledKernelOp only on CPU and no other device. We just
    // need to make sure we use and count the right one for moduleIndex
    const auto hostDevice = DeviceRefAttr::get(ctx, kCPULabel, 0);

    // Collect the chains for all module load and runtime creates.
    SmallVector<Value> moduleLoadChains;

    // Track the RuntimeCreateOp
    // Mainly used to make sure we target the right module load and device
    // runtime for the executions
    MGP::RuntimeCreateOp lastDeviceRuntimeOp;
    uint64_t deviceModuleIndex;
    for (auto op : init.getOps<MGP::RuntimeCreateOp>()) {
      moduleLoadChains.push_back(op.getResult());
      lastDeviceRuntimeOp = op;
      deviceModuleIndex = 0;
    }
    deviceContexts.init(model);

    llvm::StringMap<size_t> deviceKernelIndices;
    SmallVector<StringAttr> deviceKernelNameAttrs;
    MGP::ModuleLoadOp loadArchive;
    for (auto loadKernels : llvm::make_early_inc_range(
             init.getOps<MOGG::LoadCompiledKernelOp>())) {
      // Determine the kernelIndices and kernelNameAttrs
      deviceKernelIndices = llvm::StringMap<size_t>();
      deviceKernelNameAttrs = SmallVector<StringAttr>();
      for (StringAttr attr :
           loadKernels.getSymbols().getAsRange<StringAttr>()) {
        deviceKernelIndices[attr] = deviceKernelIndices.size();
        deviceKernelNameAttrs.push_back(attr);
      }
    }

    // Rewrite the MOGG::LoadCompiledKernelOp in init blocks.
    for (auto loadKernels : llvm::make_early_inc_range(
             init.getOps<MOGG::LoadCompiledKernelOp>())) {
      Operation *insertionOp = lastDeviceRuntimeOp.getOperation();
      size_t runtimeSlot = lastDeviceRuntimeOp.getSlot();
      Value opContext = lastDeviceRuntimeOp.getTheContext();
      Value inChain = lastDeviceRuntimeOp.getResult();

      // Rewrite the MOGG::LoadCompiledKernelOp into a MGP::ModuleLoadOp
      builder.setInsertionPointAfter(insertionOp);
      loadArchive = builder.create<MGP::ModuleLoadOp>(
          insertionOp->getLoc(),
          /*inChain=*/inChain,
          /*theContext=*/opContext,
          /*aSlot=*/runtimeSlot,
          /*bArchive=*/loadKernels.getValue(),
          /*cKernelNames=*/
          MultiLineStringAttr::get(ctx, deviceKernelNameAttrs),
          /*dConstantSymbols=*/
          MultiLineStringAttr::get(ctx, SmallVector<StringAttr>()),
          /*eDevice=*/hostDevice,
          /*fModuleIndex=*/deviceModuleIndex,
          /*gDebuginfo=*/loadKernels.getDebuginfo());
      moduleLoadChains.push_back(loadArchive.getResult());
      inChain.replaceAllUsesExcept(loadArchive.getResult(), loadArchive);
      loadKernels.erase();
    }

    // Remove the placeholder chain.
    builder.setInsertionPointToStart(&init);
    for (auto op : llvm::make_early_inc_range(
             init.getOps<MGP::ModuleLoadPlaceholderOp>())) {

      Value newChain;
      if (moduleLoadChains.size() > 1) {
        newChain =
            builder.create<GRT::ChainCreateOp>(op.getLoc(), moduleLoadChains);
      } else {
        newChain = moduleLoadChains[0];
      }
      op.replaceAllUsesWith(newChain);
      op.erase();
    }

    // Rewrite the MOGG::ExecuteCompiledOp in both init and execute blocks.
    // Shunt all the executes for Init stage at the very end of init
    builder.setInsertionPoint(init.getTerminator());
    init.walk([&](MOGG::ExecuteCompiledOp kernel) {
      auto device = kernel.getDeviceRef();
      Value inDeviceContext =
          deviceContexts.getDeviceContext(device, init.getParent());
      bool isCreator = inDeviceContext.getDefiningOp() == kernel;
      if (isCreator)
        inDeviceContext = deviceContexts.getDeviceContextFromInitArgs(device);
      rewriteExecuteCompiledOp(builder, lastDeviceRuntimeOp.getTheContext(),
                               lastDeviceRuntimeOp.getSlot(), deviceModuleIndex,
                               deviceKernelIndices[kernel.getKernelName()],
                               kernel, inDeviceContext);
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
      Value inDeviceContext = deviceContexts.getDeviceContext(
          kernel.getDeviceRef(), execute.getParent());
      rewriteExecuteCompiledOp(builder, executeContext,
                               lastDeviceRuntimeOp.getSlot(), deviceModuleIndex,
                               deviceKernelIndices[kernel.getKernelName()],
                               kernel, inDeviceContext);
    });
    // The introduction of generic execute may introduce unbound free-values.
    // Use fixupIsolatedFromAbove to rebind these.
    (void)fixupIsolatedFromAbove(builder, model);
    model.erase();
  }
}

} // namespace

} // namespace M::MGP
