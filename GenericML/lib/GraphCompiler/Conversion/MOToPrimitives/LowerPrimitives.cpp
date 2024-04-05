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

#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_LOWERPRIMITIVES
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

//===----------------------------------------------------------------------===//
// MOGraphLowering implementation
//===----------------------------------------------------------------------===//

namespace M::MGP {

namespace {

struct LowerPrimitivesPass
    : public MGP::impl::LowerPrimitivesBase<LowerPrimitivesPass> {
  using LowerPrimitivesBase::LowerPrimitivesBase;

  void runOnOperation() override;
};

void LowerPrimitivesPass::runOnOperation() {
  ModuleOp mod = getOperation();
  MLIRContext *ctx = mod->getContext();

  MOGG::MojoLibraryAnalysis mojoLibraryAnalysis(ctx);
  if (mojoLibraryAnalysis.init({}).isError())
    return signalPassFailure();

  mod.walk([&](Operation *op) {
    Location opLoc = op->getLoc();
    OpBuilder builder{op};

    // If the MGP primitive has a mojo function backing it we can lower it
    // to that function.
    if (mojoLibraryAnalysis.hasRegisteredMojoKernelFor(
            op->getName().getStringRef())) {
      // Create the MOGG generic region which contains the compiled MGP.
      SmallVector<InOutSignatureAttr::InOutSemantics> operandSemantics;
      auto newKernel = builder.create<MOGG::GenericCompiledRegionOp>(
          opLoc, op->getResultTypes(), op->getOperands(),
          KGEN::ParamDeclArrayAttr::get(ctx, {}),
          InOutSignatureAttr::get(ctx, operandSemantics), BoolAttr{}, // isKGEN
          StringAttr{},                                               // summary
          BoolAttr{},                                                 // isAsync
          BoolAttr{} // isFuseable
      );

      mlir::Block &kernelBody = newKernel.getBody().emplaceBlock();
      builder.setInsertionPoint(&kernelBody, kernelBody.begin());

      SmallVector<Value> newOperands;
      for (Value operand : newKernel->getOperands()) {
        Value val = kernelBody.addArgument(operand.getType(), operand.getLoc());
        newOperands.push_back(val);
      }

      // Add the call to the actual implementing function in mojo.
      DictionaryAttr dict = builder.getDictionaryAttr(op->getAttrs());
      MOGG::CallOp call = builder.create<MOGG::CallOp>(
          opLoc, op->getResultTypes(), op->getName().getStringRef(),
          newOperands, ArrayAttr{}, dict);

      builder.create<MOGG::OutputOp>(opLoc, call->getResults());

      op->replaceAllUsesWith(newKernel);
      op->erase();
    }
  });
}

} // namespace

} // namespace M::MGP