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

#define GEN_PASS_DEF_LOWERMOGGCOMPILES
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

//===----------------------------------------------------------------------===//
// MOGraphLowering implementation
//===----------------------------------------------------------------------===//

namespace M::MGP {

namespace {

struct LowerMOGGCompilesPass
    : public MGP::impl::LowerMOGGCompilesBase<LowerMOGGCompilesPass> {
  using LowerMOGGCompilesBase::LowerMOGGCompilesBase;

  void runOnOperation() override;
};

void LowerMOGGCompilesPass::runOnOperation() {

  MLIRContext *ctx = getOperation().getContext();

  for (auto model : getOperation().getOps<MGP::ModelOp>()) {

    Block &init = *model.getInitBlock();
    Block &execute = *model.getExecuteBlock();

    // Get the kernel creation context.
    MGP::RuntimeCreateOp kernelCtx;
    for (auto contextOp : init.getOps<MGP::RuntimeCreateOp>())
      kernelCtx = contextOp;

    uint64_t slot = kernelCtx.getSlot();

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
          /*aSlot=*/slot,
          /*bArchive=*/loadKernels.getValue(),
          /*cKernelNames=*/MultiLineStringAttr::get(ctx, kernelNameAttrs),
          /*dConstantSymbols*/
          MultiLineStringAttr::get(ctx, SmallVector<StringAttr>()),
          /*device*/ kernelCtx.getDevice());
      // Because there's no graph op, we can't use the getHostDeviceAny method.
      // Therefore I explicitly construct a new device ref with ID 0.
      // This could cause problems, but I don't know how to access devices from
      // a plain model op

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
          kernel.getOperands(), slot, 0, kernelIndices[kernel.getKernelName()]);

      kernel.replaceAllUsesWith(newExecuteOp);
      kernel.erase();
    });
  }
}

} // namespace

} // namespace M::MGP
