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

#define GEN_PASS_DEF_OUTLINEGENERICCOMPILEDREGIONSTOFUNCTIONS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

namespace M::MOGG {

namespace {

struct OutlineGenericCompiledRegionsToFunctions
    : public MOGG::impl::OutlineGenericCompiledRegionsToFunctionsBase<
          OutlineGenericCompiledRegionsToFunctions> {
  using OutlineGenericCompiledRegionsToFunctionsBase::
      OutlineGenericCompiledRegionsToFunctionsBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    OpBuilder builder(moduleOp->getContext());
    builder.setInsertionPointToStart(moduleOp.getBody(0));
    auto ctx = moduleOp->getContext();

    // The pass manager never invalidates the cached `SymbolTableAnalysis`, so
    // register new stubs in it (and mark it preserved); otherwise passes that
    // read the cached table would not see them.
    SymbolTable &symtab =
        getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

    uint64_t stubNumber = 0;
    moduleOp.walk<mlir::WalkOrder::PreOrder>(
        [&](MOGG::GenericCompiledRegionOp kernelOp) {
          auto loc = kernelOp.getLoc();
          auto resultTypes = kernelOp->getResultTypes();
          // The MGP-ABI stub takes the region body verbatim, so the signature
          // uses its `!mogg.async<T>` block arg types (the void** runtime
          // slots), not the region's `T` operand types.
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
          DeviceRefAttr device = MO::getDeviceOrDefaultAny(kernelOp);
          auto stubOp = MOGG::KernelStubOp::create(
              builder, loc, stubName, functionType, fusionInfo,
              kernelOp.getIsAsyncAttr(), kernelOp.getConstantResourcesAttr(),
              MOGG::ABIKindAttr::get(ctx, MOGG::ABIKind::MGP),
              builder.getStringAttr(device.getLabel()), nullptr, nullptr);
          stubOp.getBody().takeBody(kernelOp.getBody());

          // `create` already linked the stub into the module; this updates the
          // table's name map, which may unique the name.
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
          callOp.setDeviceRef(device);

          kernelOp->replaceAllUsesWith(callOp);
          kernelOp.erase();
          builder.setInsertionPoint(stubOp);

          return mlir::WalkResult::skip();
        });
  }
};

} // namespace

} // namespace M::MOGG
