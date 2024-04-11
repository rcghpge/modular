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
#include "GraphRT/MIPDialect/MIPDialect.h"

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

  SmallVector<Type> getMoggCallResultTypes(Operation *op);
  DictionaryAttr getMoggCallAttrs(OpBuilder &builder, Operation *op);
  void runOnOperation() override;
};

// For staging, we currently only enable fusion of MIP ops in generic regions
// all other primitives will not be fused.
static inline bool isFuseable(Operation *op) {
  return isa<::M::MIP::MIPDialect>(op->getDialect());
}

// We only allow lowering to generic compiled regions for ops which are
// registered in mojo and are are CPU only.
// FIXME: (37088) This is meant to be stop-gap as we expect generic execution
//        to follow mogg.execute dispatch between cpu/gpu in mojo, but currently
//        the generic regions are only on cpu.
static inline bool
isLowerableToGenericRegion(MLIRContext *ctx,
                           MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                           Operation *op) {
  bool isRegistered = mojoLibraryAnalysis.hasRegisteredMojoKernelFor(
      op->getName().getStringRef());
  if (auto hasDeviceOp = dyn_cast<MGPHasDeviceOpInterface>(op)) {
    return isRegistered &&
           (hasDeviceOp.getDevice() == DeviceRefAttr::get(ctx, kCPULabel, 0));
  }
  return isRegistered;
}

// For staging, we currently only enable fusion of MIP ops in generic regions
// all other primitives will not be fused.
static inline bool isLowerableType(Type type) {
  bool result = true;
  if (isa<MEF::MEFChainTypeInterface, MGP::ContextType>(type))
    result = false;
  return result;
}

// Get a vector of the result types for the lowered mogg call
SmallVector<Type> LowerPrimitivesPass::getMoggCallResultTypes(Operation *op) {
  // Chain types are a no-op within a compiled region, so they are dropped
  // from the return type in a `mogg.call`.  The chain type is preserved
  // in the output of the generic-region
  SmallVector<Type> result_types;
  for (auto resTy : op->getResultTypes())
    if (isLowerableType(resTy))
      result_types.push_back(resTy);
  return result_types;
}

// Get a dictionary of the attributes for the lowered mogg call
DictionaryAttr LowerPrimitivesPass::getMoggCallAttrs(OpBuilder &builder,
                                                     Operation *op) {
  SmallVector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    if (auto devRefAttr = dyn_cast<DeviceRefAttr>(attr.getValue())) {
      // Prune device attributes to string ref of label
      attrs.push_back(builder.getNamedAttr(
          attr.getName(), builder.getStringAttr(devRefAttr.getLabel())));
    } else {
      attrs.push_back(attr);
    }
  }
  return builder.getDictionaryAttr(attrs);
}

void LowerPrimitivesPass::runOnOperation() {
  ModuleOp mod = getOperation();
  MLIRContext *ctx = mod->getContext();

  MOGG::MojoLibraryAnalysis mojoLibraryAnalysis(ctx);
  if (mojoLibraryAnalysis.init({}).isError())
    return signalPassFailure();

  mod.walk([&](Operation *op) {
    Location opLoc = op->getLoc();
    OpBuilder builder{op};

    if (isLowerableToGenericRegion(ctx, mojoLibraryAnalysis, op)) {
      // Create the MOGG generic region which contains the compiled MGP.
      SmallVector<InOutSignatureAttr::InOutSemantics> operandSemantics;
      auto newKernel = builder.create<MOGG::GenericCompiledRegionOp>(
          opLoc, op->getResultTypes(), op->getOperands(),
          KGEN::ParamDeclArrayAttr::get(ctx, {}),
          InOutSignatureAttr::get(ctx, operandSemantics), BoolAttr{}, // isKGEN
          StringAttr{},                                               // summary
          BoolAttr{},                                                 // isAsync
          BoolAttr::get(ctx, isFuseable(op)) // isFuseable
      );

      mlir::Block &kernelBody = newKernel.getBody().emplaceBlock();
      builder.setInsertionPoint(&kernelBody, kernelBody.begin());

      SmallVector<Value> newOperands;
      for (Value operand : newKernel->getOperands()) {
        Value val = kernelBody.addArgument(operand.getType(), operand.getLoc());
        if (isLowerableType(operand.getType())) {
          newOperands.push_back(val);
        }
      }

      // Add the call to the actual implementing function in mojo.
      MOGG::CallOp call = builder.create<MOGG::CallOp>(
          opLoc, getMoggCallResultTypes(op), op->getName().getStringRef(),
          newOperands, ArrayAttr{}, getMoggCallAttrs(builder, op));

      builder.create<MOGG::OutputOp>(opLoc, call->getResults());

      op->replaceAllUsesWith(newKernel);
      op->erase();
    }
  });
}

} // namespace

} // namespace M::MGP