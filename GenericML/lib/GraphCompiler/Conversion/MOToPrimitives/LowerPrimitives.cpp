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
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "GraphRT/MIPDialect/MIPDialect.h"
#include "Support/MDialect/MAttrs.h"

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

  DictionaryAttr getMoggCallAttrs(OpBuilder &builder, Operation *op);
  void runOnOperation() override;
};

// For staging, we currently only enable fusion of MIP ops in generic regions
// all other primitives will not be fused.
static inline bool canFuseIntoOther(Operation *op) {
  return isa<::M::MIP::MIPDialect, MGPDialect>(op->getDialect());
}

static inline bool canDuplicate(Operation *op) {
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
    // DeviceContextCreateOp uses the gpu device context but not the gpu. Allow
    // it to be lowered.
    return isRegistered && ((hasDeviceOp.getDevice() ==
                             DeviceRefAttr::get(ctx, kCPULabel, 0)) ||
                            dyn_cast<MGP::DeviceContextCreateOp>(op));
  }
  return isRegistered;
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
    } else if (auto shapeAttr =
                   dyn_cast<IntArrayElementsAttr>(attr.getValue())) {
      // For Static TensorSpec attributes, we want to generate the rank param
      // and pass the static shape.
      auto shape = shapeAttr.asArrayRef<int64_t>();
      auto name = attr.getName().str();
      attrs.push_back(builder.getNamedAttr(name + "Rank",
                                           builder.getIndexAttr(shape.size())));
      attrs.push_back(
          builder.getNamedAttr(name, builder.getI64TensorAttr(shape)));
    } else {
      attrs.push_back(attr);
    }
  }
  // For anything that takes in a tensor-spec input, we need to parameterize the
  // rank so the mojo calls can use that.
  for (auto operandType : op->getOperands().getTypes()) {
    if (auto tensorSpec = dyn_cast<MGP::TensorSpecType>(operandType)) {
      attrs.push_back(builder.getNamedAttr(
          "rank", builder.getIndexAttr(tensorSpec.getShape().size())));
    }
  }

  return builder.getDictionaryAttr(attrs);
}

void LowerPrimitivesPass::runOnOperation() {
  ModuleOp mod = getOperation();
  MLIRContext *ctx = mod->getContext();

  auto &analysis = getAnalysis<MOGG::LoadMojoAnalysisPass>();
  ErrorOr<MOGG::MojoLibraryAnalysis *> errOr =
      analysis.initIfNeeded(ctx, kernelLibraryPaths);
  if (errOr.isError())
    return signalPassFailure();
  MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis = *errOr.takeValue();

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
          StringAttr::get(ctx, op->getName().getStringRef()),         // summary
          BoolAttr{},                                                 // isAsync
          BoolAttr::get(ctx, canFuseIntoOther(op)), // canFuseIntoOther
          BoolAttr::get(ctx, canDuplicate(op))      // canDuplicate
      );

      mlir::Block &kernelBody = newKernel.getBody().emplaceBlock();
      builder.setInsertionPoint(&kernelBody, kernelBody.begin());

      for (Value operand : newKernel->getOperands())
        kernelBody.addArgument(operand.getType(), operand.getLoc());

      // Add the call to the actual implementing function in mojo.
      MOGG::CallOp call = builder.create<MOGG::CallOp>(
          opLoc, op->getResultTypes(), op->getName().getStringRef(),
          kernelBody.getArguments(), ArrayAttr{},
          getMoggCallAttrs(builder, op));

      builder.create<MOGG::OutputOp>(opLoc, call->getResults());

      op->replaceAllUsesWith(newKernel);
      op->erase();
    }
  });
}

} // namespace

} // namespace M::MGP
