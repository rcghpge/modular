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

  void runOnOperation() override;
};

// Take all the compiled JIT Kernels and lower them
// -- Lower MOGG::LoadCompiledKernelOp to MGP::ModuleLoadOp
//    1. Determine the potential insertion point based off following rules
//       a. If there is ModuleLoadOp for the device --> After ModuleLoadOp +
//       keep track of Module Index for the host device (skipping non-host)
//       b. If no ModuleLoadOp for the specific device --> After RuntimeCreateOp
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

    // MOGG::LoadCompiledKernelOp only on CPU and no other device. We just need
    // to make sure we use and count the right one for moduleIndex
    const auto device = DeviceRefAttr::get(ctx, kCPULabel, 0);

    // Track the RuntimeCreateOp and ModuleLoadOp and their last use
    llvm::SmallDenseMap<DeviceRefAttr, MGP::RuntimeCreateOp>
        lastDeviceRuntimeOp;
    llvm::SmallDenseMap<DeviceRefAttr, MGP::ModuleLoadOp>
        lastDeviceModuleLoadOp;
    llvm::SmallDenseMap<DeviceRefAttr, uint64_t> deviceModuleIndex;
    for (auto op : init.getOps<MGP::RuntimeCreateOp>()) {
      lastDeviceRuntimeOp[op.getDevice()] = op;
      deviceModuleIndex[op.getDevice()] = 0;
    }
    for (auto op : init.getOps<MGP::ModuleLoadOp>()) {
      lastDeviceModuleLoadOp[op.getDevice()] = op;
      GENERICML_ASSERT(deviceModuleIndex.contains(op.getDevice()))
          << "Cannot have a ModuleLoadOp for a device: " << op.getDevice()
          << " without a corresponding RuntimeCreateOp";
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

      // Replace the loadKernels Op
      if (lastDeviceModuleLoadOp.contains(device)) {
        builder.setInsertionPointAfter(lastDeviceModuleLoadOp[device]);
        loadArchive = builder.create<MGP::ModuleLoadOp>(
            lastDeviceModuleLoadOp[device].getLoc(),
            /*inChain=*/lastDeviceModuleLoadOp[device].getResult(),
            /*theContext=*/lastDeviceModuleLoadOp[device].getTheContext(),
            /*aSlot=*/lastDeviceModuleLoadOp[device].getSlot(),
            /*bArchive=*/loadKernels.getValue(),
            /*cKernelNames=*/
            MultiLineStringAttr::get(ctx, deviceKernelNameAttrs[device]),
            /*dConstantSymbols*/
            MultiLineStringAttr::get(ctx, SmallVector<StringAttr>()),
            /*device*/ device);
        lastDeviceModuleLoadOp[device].getResult().replaceAllUsesExcept(
            loadArchive.getResult(), loadArchive);
      } else {
        builder.setInsertionPointAfter(lastDeviceRuntimeOp[device]);
        loadArchive = builder.create<MGP::ModuleLoadOp>(
            lastDeviceRuntimeOp[device].getLoc(),
            /*inChain=*/lastDeviceRuntimeOp[device].getResult(),
            /*theContext=*/lastDeviceRuntimeOp[device].getTheContext(),
            /*aSlot=*/lastDeviceRuntimeOp[device].getSlot(),
            /*bArchive=*/loadKernels.getValue(),
            /*cKernelNames=*/
            MultiLineStringAttr::get(ctx, deviceKernelNameAttrs[device]),
            /*dConstantSymbols*/
            MultiLineStringAttr::get(ctx, SmallVector<StringAttr>()),
            /*device*/ device);
        lastDeviceRuntimeOp[device].getResult().replaceAllUsesExcept(
            loadArchive.getResult(), loadArchive);
      }
      loadKernels.erase();
    }

    // Shunt all the executes for Init stage at the very end of init
    MEF::OutputOp mefOutputOp;
    for (auto op : init.getOps<MEF::OutputOp>())
      mefOutputOp = op;
    init.walk([&](MOGG::ExecuteCompiledOp kernel) {
      auto loc = kernel.getLoc();
      builder.setInsertionPoint(mefOutputOp);
      auto newExecuteOp = builder.create<MGP::ExecuteGenericOp>(
          loc, kernel.getResultTypes(),
          lastDeviceRuntimeOp[device].getTheContext(), kernel.getOperands(),
          lastDeviceRuntimeOp[device].getSlot(), deviceModuleIndex[device],
          deviceKernelIndices[device][kernel.getKernelName()]);
      kernel.replaceAllUsesWith(newExecuteOp);
      kernel.erase();
    });

    Value executeContext = nullptr;
    for (Value arg : execute.getArguments()) {
      if (isa<MGP::ContextType>(arg.getType()))
        executeContext = arg;
    }
    execute.walk([&](MOGG::ExecuteCompiledOp kernel) {
      builder.setInsertionPoint(kernel);
      auto newExecuteOp = builder.create<MGP::ExecuteGenericOp>(
          kernel.getLoc(), kernel.getResultTypes(), executeContext,
          kernel.getOperands(), lastDeviceRuntimeOp[device].getSlot(),
          deviceModuleIndex[device],
          deviceKernelIndices[device][kernel.getKernelName()]);

      kernel.replaceAllUsesWith(newExecuteOp);
      kernel.erase();
    });
  }
}

} // namespace

} // namespace M::MGP
