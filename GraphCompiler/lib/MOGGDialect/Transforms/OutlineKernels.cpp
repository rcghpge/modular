//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Interfaces/LoweringState.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGInterfaces.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/SymbolTable.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_OUTLINEKERNELS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

namespace M::MOGG {

namespace {

struct OutlineKernels : public MOGG::impl::OutlineKernelsBase<OutlineKernels> {
  using OutlineKernelsBase::OutlineKernelsBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    OpBuilder builder(moduleOp->getContext());
    builder.setInsertionPointToStart(moduleOp.getBody(0));
    auto ctx = moduleOp->getContext();

    // Register newly outlined kernel stubs through the module's cached
    // `SymbolTableAnalysis` so it stays consistent for any downstream consumer.
    // The analysis is never invalidated by the pass manager, so a stub added
    // only via the builder would be invisible to passes that read the cached
    // table. We keep it up to date here, hence we can mark it preserved.
    SymbolTable &symtab =
        getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

    uint64_t stubNumber = 0;
    moduleOp.walk<mlir::WalkOrder::PreOrder>(
        [&](MOGG::GenericCompiledRegionOp kernelOp) {
          auto loc = kernelOp.getLoc();
          auto resultTypes = kernelOp->getResultTypes();
          // The MGP-ABI stub takes the region body verbatim, whose block
          // arguments are `!mogg.async<T>` (the void** runtime slots). The
          // function signature must match those body argument types, not the
          // region's `T` operand types.
          auto functionType = TypeAttr::get(
              FunctionType::get(ctx, kernelOp.getBodyBlock().getArgumentTypes(),
                                kernelOp.getResultTypes()));
          auto stubName =
              StringAttr::get(kernelOp->getContext(),
                              llvm::formatv("region_{}", stubNumber++).str());
          // Synthesize a leaf FusionInfoAttr from the summary when the region
          // has no fusionInfo, so KernelStubOp always carries one.
          FusionInfoAttr fusionInfo = kernelOp.getFusionInfoAttr();
          if (!fusionInfo) {
            StringAttr opName = stubName;
            if (auto s = kernelOp->getAttrOfType<StringAttr>("summary"))
              opName = s;
            fusionInfo = FusionInfoAttr::get(
                ctx, FusionLeafAttr::get(ctx, opName, /*loc=*/{}, /*sig=*/{}));
          }
          auto stubOp = MOGG::KernelStubOp::create(
              builder, loc, stubName, functionType, fusionInfo,
              kernelOp.getIsAsyncAttr(), kernelOp.getConstantResourcesAttr(),
              MOGG::ABIKindAttr::get(ctx, MOGG::ABIKind::MGP), nullptr,
              nullptr);
          stubOp.getBody().takeBody(kernelOp.getBody());
          DeviceRefAttr device = MO::getDeviceOrDefaultAny(kernelOp);
          stubOp.setDeviceRef(device);

          // Register the stub in the cached symbol table. `create` already
          // linked it into the module, so this only updates the table's name
          // map; reuse the returned name for the call site in case of uniquing.
          stubName = symtab.insert(stubOp);

          builder.setInsertionPoint(kernelOp);
          auto callOp = MOGG::ExecuteCompiledOp::create(
              builder, stubOp->getLoc(),
              /*resultTypes=*/resultTypes,
              /*operands=*/kernelOp->getOperands(),
              /*kernelName=*/FlatSymbolRefAttr::get(stubName),
              /*usedParams=*/builder.getArrayAttr({}),
              /*outputParamDecls=*/
              KGEN::ParamDeclArrayAttr::get(builder.getContext(), {}),
              /*fusionInfo=*/stubOp.getFusionInfoAttr(),
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

} // namespace M::MOGG
