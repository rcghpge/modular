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

void LowerMOGGRegionCompilesPass::runOnOperation() {

  MLIRContext *ctx = getOperation().getContext();

  for (auto model : getOperation().getOps<MGP::ModelOp>()) {

    Block &init = *model.getInitBlock();
    Block &execute = *model.getExecuteBlock();

    // Get the kernel creation context.
    MGP::RuntimeCreateOp kernelCtx;
    for (auto contextOp : init.getOps<MGP::RuntimeCreateOp>())
      kernelCtx = contextOp;
    uint64_t runtimeSlot = kernelCtx.getSlot();

    // Get the existing module load if it already exists
    // TODO: (#34789) Need to remove extra layer of moduleIndex once we have 1
    // mogg compile for all kernels
    uint64_t moduleIndex = llvm::range_size(init.getOps<MGP::ModuleLoadOp>());

    OpBuilder builder{ctx};
    llvm::StringMap<size_t> kernelIndices;
    for (auto loadKernels : llvm::make_early_inc_range(
             init.getOps<MOGG::LoadCompiledKernelOp>())) {
      SmallVector<StringAttr> kernelNameAttrs;

      for (StringAttr attr :
           loadKernels.getSymbols().getAsRange<StringAttr>()) {
        kernelIndices[attr] = kernelIndices.size();
        kernelNameAttrs.push_back(attr);
      }

      builder.setInsertionPointAfter(kernelCtx);

      auto loadArchive = builder.create<MGP::ModuleLoadOp>(
          kernelCtx.getLoc(),
          /*inChain=*/kernelCtx.getResult(),
          /*theContext=*/kernelCtx.getTheContext(),
          /*aSlot=*/runtimeSlot,
          /*bArchive=*/loadKernels.getValue(),
          /*cKernelNames=*/MultiLineStringAttr::get(ctx, kernelNameAttrs),
          /*dConstantSymbols*/
          MultiLineStringAttr::get(ctx, SmallVector<StringAttr>()),
          /*device*/ kernelCtx.getDevice());

      kernelCtx.getResult().replaceAllUsesExcept(loadArchive.getResult(),
                                                 loadArchive);
      loadKernels.erase();
    }

    Value executeContext = nullptr;
    for (Value arg : execute.getArguments()) {
      if (isa<MGP::ContextType>(arg.getType()))
        executeContext = arg;
    }

    execute.walk([&](MOGG::ExecuteCompiledOp kernel) {
      builder.setInsertionPoint(kernel);
      auto newExecuteOp = builder.create<MGP::ExecuteGenericOp>(
          kernel.getLoc(), kernel.getResultTypes(), executeContext,
          kernel.getOperands(), runtimeSlot, moduleIndex,
          kernelIndices[kernel.getKernelName()]);

      kernel.replaceAllUsesWith(newExecuteOp);
      kernel.erase();
    });
  }
}

} // namespace

} // namespace M::MGP
