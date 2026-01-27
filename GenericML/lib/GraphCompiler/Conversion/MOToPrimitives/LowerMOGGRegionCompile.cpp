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
#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GenericML/GraphCompiler/MGPDialect/Transforms/FixupIsolatedFromAbove.h"
#include "GenericML/GraphCompiler/MGPDialect/Transforms/Utils.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
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
  void rewriteExecuteCompiledOp(OpBuilder &builder, const Value &context,
                                size_t runtimeSlot, size_t module_index,
                                size_t kernel_index,
                                MOGG::ExecuteCompiledOp &oldKernel);
  void runOnOperation() override;
};

// Rewrite the MOGG::ExecuteCompiledOp into a MGP::ExecuteGenericOp
void LowerMOGGRegionCompilesPass::rewriteExecuteCompiledOp(
    OpBuilder &builder, const Value &context, size_t runtimeSlot,
    size_t module_index, size_t kernel_index,
    MOGG::ExecuteCompiledOp &oldKernel) {
  auto ctx = oldKernel.getOperation()->getContext();
  SmallVector<Value> outputs;
  SmallVector<Type> resultTypes;
  for (auto [resTy, res] :
       llvm::zip(oldKernel.getResultTypes(), oldKernel.getResults())) {
    outputs.push_back(res);
    // In emit-mojo path, convert mo.opaque result types to mgp.opaque for
    // the model's external interface. Preserve type parameters.
    auto moOpaqueType = dyn_cast<MO::OpaqueType>(resTy);
    if (!moOpaqueType) {
      resultTypes.push_back(resTy);
      continue;
    }
    resultTypes.push_back(MGP::OpaqueType::get(
        ctx, moOpaqueType.getSymbol(), moOpaqueType.getOptionalParameters()));
  }

  auto device = oldKernel.getExecutionDevice();
  if (!device)
    device = DeviceRefAttr::getDefault(ctx);

  auto newExecuteOp = MGP::ExecuteGenericOp::create(
      builder, oldKernel.getLoc(), resultTypes, context,
      oldKernel->getOperands(), runtimeSlot, module_index, kernel_index,
      oldKernel.getSummary(), oldKernel.getIsAsync(),
      oldKernel.getConstantResources());

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
    for (auto op : init.getOps<MGP::RuntimeCreateOp>()) {
      moduleLoadChains.push_back(op.getResult());
      lastDeviceRuntimeOp = op;
    }

    // For each kernel name, track the module index and kernel index
    llvm::StringMap<std::pair<uint64_t, uint64_t>> kernelModuleAndFnIndex;
    uint64_t deviceModuleIndex = 0;
    // Rewrite the MOGG::LoadCompiledKernelOp in init blocks.
    for (auto loadKernels : llvm::make_early_inc_range(
             init.getOps<MOGG::LoadCompiledKernelOp>())) {
      Operation *insertionOp = lastDeviceRuntimeOp.getOperation();
      size_t runtimeSlot = lastDeviceRuntimeOp.getSlot();
      Value opContext = lastDeviceRuntimeOp.getTheContext();
      Value inChain = lastDeviceRuntimeOp.getResult();

      SmallVector<StringAttr> deviceKernelNameAttrs;
      uint64_t deviceKernelIndex = 0;
      for (StringAttr attr :
           loadKernels.getSymbols().getAsRange<StringAttr>()) {
        kernelModuleAndFnIndex[attr] =
            std::make_pair(deviceModuleIndex, deviceKernelIndex++);
        deviceKernelNameAttrs.push_back(attr);
      }

      // Rewrite the MOGG::LoadCompiledKernelOp into a MGP::ModuleLoadOp
      builder.setInsertionPointAfter(insertionOp);
      auto loadArchive = MGP::ModuleLoadOp::create(
          builder, insertionOp->getLoc(),
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
      ++deviceModuleIndex;
    }

    // Remove the placeholder chain.
    builder.setInsertionPointToStart(&init);
    for (auto op : llvm::make_early_inc_range(
             init.getOps<MGP::ModuleLoadPlaceholderOp>())) {

      Value newChain;
      if (moduleLoadChains.size() > 1) {
        newChain =
            GRT::ChainCreateOp::create(builder, op.getLoc(), moduleLoadChains);
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
      auto [moduleIndex, kernelIndex] =
          kernelModuleAndFnIndex[kernel.getCalleeSymbol()];
      rewriteExecuteCompiledOp(builder, lastDeviceRuntimeOp.getTheContext(),
                               lastDeviceRuntimeOp.getSlot(), moduleIndex,
                               kernelIndex, kernel);
    });

    execute.walk([&](MOGG::ExecuteCompiledOp kernel) {
      builder.setInsertionPoint(kernel);
      Region *parentRegion = kernel->getParentRegion();
      Value executeContext = nullptr;

      // For execute block, need to account for the bounding region in the
      // case of control flow or other bounding operations.
      while (parentRegion && !executeContext) {
        for (Value arg : parentRegion->getArguments()) {
          if (isa<MGP::ContextType>(arg.getType()))
            executeContext = arg;
        }
        parentRegion = parentRegion->getParentRegion();
      }
      ASSERT_STREAM(executeContext,
                    << "Cannot find execute context in bounding region of "
                       "generic.execute");
      auto [moduleIndex, kernelIndex] =
          kernelModuleAndFnIndex[kernel.getCalleeSymbol()];
      rewriteExecuteCompiledOp(builder, executeContext,
                               lastDeviceRuntimeOp.getSlot(), moduleIndex,
                               kernelIndex, kernel);
    });

    // In emit-mojo path, ToMGPFromOpaqueOp is used to convert mo.opaque to
    // mgp.opaque for graph outputs. At this point, the Mojo code has already
    // been emitted and handles the value correctly. Remove the op by replacing
    // with its input since the runtime doesn't need this type conversion.
    for (auto toMgpOp : llvm::make_early_inc_range(
             execute.getOps<MOGG::ToMGPFromOpaqueOp>())) {
      toMgpOp.getDataPointer().replaceAllUsesWith(toMgpOp.getInput());
      toMgpOp.erase();
    }
    // The introduction of generic execute may introduce unbound free-values.
    // Use fixupIsolatedFromAbove to rebind these.
    (void)fixupIsolatedFromAbove(builder, model);
    model.erase();
  }
}

} // namespace

} // namespace M::MGP
