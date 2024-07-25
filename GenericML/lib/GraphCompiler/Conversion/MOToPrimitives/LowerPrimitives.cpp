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

// All MIP and MGP Primitives that are registered we will allow fusion of.
static inline bool canFuseOp(Operation *op) {
  // Allow for custom ops starting with "testfuse." for testing purposes.
  return isa_and_nonnull<::M::MIP::MIPDialect, MGPDialect>(op->getDialect()) ||
         op->getName().getIdentifier().strref().starts_with("testfuse.");
}

static inline bool canDuplicate(Operation *op) {
  // Allow for custom ops with "testfuse" for testing purposes.
  if (op->getName().getIdentifier().strref().starts_with("testfuse."))
    return true;

  // TODO(GRA-784): This is disabled since certain subgraphs can bloat
  // compile times.
  // return isa<::M::MIP::MIPDialect>(op->getDialect());
  return false;
}

// Anything we find a registry for, lower to generic regions
static inline bool
isLowerableToGenericRegion(MLIRContext *ctx,
                           MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                           Operation *op) {
  bool isRegistered = mojoLibraryAnalysis.hasRegisteredMojoKernelFor(
      op->getName().getStringRef());
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
      bool hasCuda = false;
      if (auto hasDeviceOp = dyn_cast<MGPHasDeviceOpInterface>(op))
        hasCuda = hasDeviceOp.getDevice().getLabel() == kCUDALabel;
      auto newKernel = builder.create<MOGG::GenericCompiledRegionOp>(
          opLoc, op->getResultTypes(), op->getOperands(),
          KGEN::ParamDeclArrayAttr::get(ctx, {}),
          InOutSignatureAttr::get(ctx, operandSemantics), BoolAttr{}, // isKGEN
          StringAttr::get(ctx, op->getName().getStringRef()),         // summary
          BoolAttr{},                                                 // isAsync
          BoolAttr::get(ctx, hasCuda),                                // hasCuda
          BoolAttr::get(ctx, canFuseOp(op)),   // canFuseIntoOther
          BoolAttr::get(ctx, canFuseOp(op)),   // canBeFusedInto
          BoolAttr::get(ctx, canDuplicate(op)) // canDuplicate
      );

      mlir::Block &kernelBody = newKernel.getBody().emplaceBlock();
      builder.setInsertionPoint(&kernelBody, kernelBody.begin());

      llvm::SmallDenseMap<Value, Value> operandToArgMap;
      for (Value operand : newKernel->getOperands()) {
        operandToArgMap[operand] =
            kernelBody.addArgument(operand.getType(), operand.getLoc());
      }

      KGEN::GeneratorOp kernel = nullptr;

      // Convert opaque inputs.
      SmallVector<Value> inputs(kernelBody.getArguments());
      for (auto [idx, ty] : llvm::enumerate(newKernel.getOperandTypes())) {
        if (!isa<MGP::OpaqueType>(ty))
          continue;

        // Need to add extra ops for opaque outputs.
        // Load the type infos from the kernel.
        if (!kernel)
          kernel = mojoLibraryAnalysis.getRegisteredMojoKernelFor(
              op->getName().getStringRef());
        ASSERT_STREAM(kernel != nullptr, << "kernel `"
                                         << op->getName().getStringRef()
                                         << "` not found");
        bool isRegisterType = kernel.getSignature().getArgConvention(idx) ==
                              KGEN::ArgConvention::BorrowedInReg;

        // Convert MGP value to opaque.
        // TODO: Simplify mgp.opaque lowerings for MGP Primitives (GRA-812).
        Value val = inputs[idx];
        auto moOpaqueTy = MO::OpaqueType::get(
            kernel->getContext(),
            StringAttr::get(kernel->getContext(), "OpaquePrimitive"),
            std::nullopt);
        val = builder.create<MOGG::ToOpaqueFromMGPOp>(opLoc, moOpaqueTy, val);
        inputs[idx] = builder.create<MOGG::ToKGENOp>(
            opLoc, kernel.getFunctionType().getInput(idx), val,
            /*shape=*/nullptr,
            /*strides=*/nullptr, /*typeMethodDict=*/nullptr, isRegisterType);
      }

      // Add the call to the actual implementing function in mojo.
      MOGG::CallOp call = builder.create<MOGG::CallOp>(
          opLoc, op->getResultTypes(), op->getName().getStringRef(), inputs,
          ArrayAttr{}, getMoggCallAttrs(builder, op));

      // Convert opaque outputs
      SmallVector<Value> outputs(call->getResults());
      for (auto [idx, ty] : llvm::enumerate(op->getResultTypes())) {
        if (!isa<MGP::OpaqueType>(ty))
          continue;

        // Need to add extra ops for opaque outputs.
        // Load the type infos from the kernel.
        if (!kernel)
          kernel = mojoLibraryAnalysis.getRegisteredMojoKernelFor(
              op->getName().getStringRef());
        ASSERT_STREAM(kernel != nullptr, << "kernel `"
                                         << op->getName().getStringRef()
                                         << "` not found");

        FailureOr<DictionaryAttr> methodsDictAttrOr =
            MOGG::buildMojoObjectInfosOutputMethodsDict(
                kernel, ArrayRef<StringRef>{"__moveinit__", "__del__"});
        if (failed(methodsDictAttrOr)) {
          op->emitError("missing methods for the primitive output opaque type: "
                        "__moveinit__ / __del__ not found");
          signalPassFailure();
          return;
        }
        bool isRegister = !kernel.getSignature().hasMemoryOnlyResult();

        // Convert opaque to MGP value.
        // TODO: Simplify mgp.opaque lowerings for MGP Primitives (GRA-812).
        Value val = outputs[idx];
        auto moOpaqueTy = MO::OpaqueType::get(
            kernel->getContext(),
            StringAttr::get(kernel->getContext(), "OpaquePrimitive"),
            std::nullopt);
        val = builder.create<MOGG::ToMOGGOp>(
            opLoc, moOpaqueTy, val, /*shape=*/nullptr, /*strides=*/nullptr,
            *methodsDictAttrOr, isRegister);
        outputs[idx] = builder.create<MOGG::ToMGPFromOpaqueOp>(
            opLoc, outputs[idx].getType(), val);
      }

      builder.create<MOGG::OutputOp>(opLoc, outputs);
      op->replaceAllUsesWith(newKernel);
      op->erase();
    }
  });
}

} // namespace

} // namespace M::MGP
