//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/Interfaces/LoweringState.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGInterfaces.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_OUTLINEKERNELS
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

namespace M::MGP {

namespace {

struct OutlineKernels : public MGP::impl::OutlineKernelsBase<OutlineKernels> {
  using OutlineKernelsBase::OutlineKernelsBase;

  void runOnOperation() override {
    if (!MOGG::useExperimentalMojoPath())
      return;

    auto moduleOp = getOperation();
    OpBuilder builder(moduleOp->getContext());
    builder.setInsertionPointToStart(moduleOp.getBody(0));
    auto ctx = moduleOp->getContext();

    uint64_t stubNumber = 0;
    moduleOp.walk<mlir::WalkOrder::PreOrder>(
        [&](MOGG::GenericCompiledRegionOp kernelOp) {
          if (!kernelOp.getEmitsMojo())
            return mlir::WalkResult::skip();

          auto loc = kernelOp.getLoc();
          auto resultTypes = kernelOp->getResultTypes();
          auto functionType = TypeAttr::get(FunctionType::get(
              ctx, kernelOp->getOperandTypes(), kernelOp.getResultTypes()));
          auto stubName =
              StringAttr::get(kernelOp->getContext(),
                              llvm::formatv("region_{}", stubNumber++).str());
          auto stubOp = MOGG::KernelStubOp::create(
              builder, loc, stubName, functionType, kernelOp.getSummaryAttr(),
              kernelOp.getIsAsyncAttr(), kernelOp.getConstantResourcesAttr(),
              MOGG::ABIKindAttr::get(ctx, MOGG::ABIKind::MGP));
          stubOp.getBody().takeBody(kernelOp.getBody());
          DeviceRefAttr device = MO::getDeviceOrDefaultAny(kernelOp);
          stubOp.setDeviceRef(device);

          builder.setInsertionPoint(kernelOp);
          auto callOp = MOGG::ExecuteCompiledOp::create(
              builder, stubOp->getLoc(),
              /*resultTypes=*/resultTypes,
              /*operands=*/kernelOp->getOperands(),
              /*kernelName=*/FlatSymbolRefAttr::get(stubName),
              /*usedParams=*/builder.getArrayAttr({}),
              /*outputParamDecls=*/
              KGEN::ParamDeclArrayAttr::get(builder.getContext(), {}),
              /*summary=*/stubOp.getSummaryAttr(),
              /*isAsync=*/stubOp.getIsAsyncAttr(),
              /*constantResources=*/stubOp.getConstantResources());
          callOp.setDeviceRef(stubOp.getExecutionDevice());

          kernelOp->replaceAllUsesWith(callOp);
          kernelOp.erase();
          builder.setInsertionPoint(stubOp);

          return mlir::WalkResult::skip();
        });
  }
};

} // namespace

} // namespace M::MGP
