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
#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/IR/IRMapping.h"

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

using CompiledRegionBodyBuilder = std::function<ErrorOr<SmallVector<Value>>(
    OpBuilder &, Location, Operation *, ValueRange)>;
struct LowerPrimitivesPass
    : public MOGG::MojoLibraryAnalysisPass<
          MGP::impl::LowerPrimitivesBase<LowerPrimitivesPass>> {
  using Base::Base;

  void runOnOperation() override;
};

// All POP, index and MGP Primitives that are registered we will allow fusion
// of.
static inline bool canFuseOp(Operation *op) {
  if (isa<MGP::SyncOp>(op))
    return false;

  // Do not fused buffer.constant.external as doing it would lead to a HUGE
  // region.
  if (isa<MGP::BufferAllocOp, MGP::BufferConstantExternalOp>(*op))
    return false;

  // Allow for custom ops starting with "testfuse." for testing purposes.
  return isa_and_nonnull<MGPDialect, KGEN::POP::POPDialect,
                         mlir::index::IndexDialect>(op->getDialect()) ||
         isa<KGEN::ParamConstantOp>(op) ||
         op->getName().getIdentifier().strref().starts_with("testfuse.");
}

static inline bool isAsync(Operation *op) {
  // Event sync should be launched in a separate thread.
  return isa<MGP::SyncOp>(op);
}

static inline bool canDuplicate(Operation *op) {
  // Allow for custom ops with "testfuse" for testing purposes.
  if (op->getName().getIdentifier().strref().starts_with("testfuse."))
    return true;

  if (isa<MGP::BufferSliceOp, MGP::BufferConstantOp, MGP::TensorSpecGetDimOp,
          MGP::BufferConstantExternalOp, MGP::BufferGetCachedOp,
          MGP::TensorSpecCreateOp, MGP::TensorExtractBufferOp,
          MGP::TensorExtractTensorSpecOp>(op))
    return true;
  return isa<KGEN::KGENDialect, KGEN::POP::POPDialect,
             mlir::index::IndexDialect>(op->getDialect());
}

// Wrap a mogg.generic_compiled_region around the given operation.
static ErrorOr<MOGG::GenericCompiledRegionOp>
wrapGenericRegionAround(OpBuilder &builder, Location loc, Operation *op,
                        CompiledRegionBodyBuilder bodyBuilder) {
  auto constResources = builder.getArrayAttr({});
  if (auto attrToValOp = dyn_cast<MGP::MGPSelectiveLowerAttrOpInterface>(op))
    constResources = builder.getArrayAttr(attrToValOp.getLiftableAttributes());

  // Build compile region operation.
  SmallVector<InOutSignatureAttr::InOutSemantics> operandSemantics(
      op->getNumOperands(), InOutSignatureAttr::InOutSemantics::kNone);
  auto newKernel = MOGG::GenericCompiledRegionOp::create(
      builder, loc, op->getResultTypes(), op->getOperands(),
      KGEN::ParamDeclArrayAttr::get(builder.getContext(), {}),
      InOutSignatureAttr::get(builder.getContext(), operandSemantics),
      MOGG::getSummaryString(op),           // summary
      builder.getBoolAttr(isAsync(op)),     // isAsync
      constResources,                       // constantResources
      builder.getBoolAttr(canFuseOp(op)),   // canFuseIntoOther
      builder.getBoolAttr(canFuseOp(op)),   // canBeFusedInto
      builder.getBoolAttr(canDuplicate(op)) // canDuplicate
  );

  if (auto hasDeviceOp = dyn_cast<MGPHasDeviceOpInterface>(op))
    newKernel.setDeviceRef(hasDeviceOp.getDevice());
  else
    newKernel.setDeviceRef(MO::getDeviceOrDefaultAny(op));

  mlir::Block &kernelBody = newKernel.getBody().emplaceBlock();
  for (Value operand : newKernel->getOperands())
    kernelBody.addArgument(operand.getType(), operand.getLoc());

  // Build the kernel body.
  builder.setInsertionPoint(&kernelBody, kernelBody.begin());
  ErrorOr<SmallVector<Value>> outputs =
      bodyBuilder(builder, loc, op, kernelBody.getArguments());
  if (outputs.isError())
    return outputs.takeError();
  SmallVector<Value> outs = outputs.takeValue();

  SmallVector<Value> returns;
  for (auto [returnVal, returnType] :
       llvm::zip(outs, newKernel.getResultTypes())) {
    if (returnVal.getType() != returnType &&
        isa<IndexType>(returnVal.getType())) {
      returnVal = mlir::index::CastSOp::create(builder, returnVal.getLoc(),
                                               returnType, returnVal);
    }
    returns.push_back(returnVal);
  }

  MOGG::OutputOp::create(builder, loc, returns);
  return newKernel;
}

void LowerPrimitivesPass::runOnOperation() {
  ModelOp model = getOperation();
  MLIRContext *ctx = model->getContext();

  auto cloneOperationBuilder =
      [](OpBuilder &builder, Location loc, Operation *op,
         ValueRange inputs) -> ErrorOr<SmallVector<Value>> {
    // Simply clone the operation into the compiled region.
    mlir::IRMapping mapping;
    for (auto [from, to] : llvm::zip_equal(op->getOperands(), inputs))
      mapping.map(from, to);
    Operation *newOp = builder.clone(*op, mapping);
    return newOp->getResults();
  };

  const auto walkerFn = [&](Operation *op) {
    // Skip any operations nested within a region.
    if (isa<MOGG::GenericCompiledRegionOp, MOGG::KernelStubOp>(op))
      return mlir::WalkResult::skip();

    // Constant external ops should not be lowered to mojo so they can be
    // updated with a prefix during inlining.
    if (isa<MGP::BufferConstantExternalOp>(op))
      return mlir::WalkResult::advance();

    /// TODO(GEX-2588): Remove this.
    if (isa<MGP::BufferCacheOp, MGP::RuntimeCreateOp, MGP::ModuleLoadOp,
            MGP::ExecuteGenericOp>(op)) {
      return mlir::WalkResult::advance();
    }

    auto implementsEmitsMojoInterface = dyn_cast<MOGG::EmitsMojoInterface>(op);
    bool isAnIndexOrPOPOrParamConstantOp =
        isa_and_nonnull<mlir::index::IndexDialect, KGEN::POP::POPDialect>(
            op->getDialect()) ||
        isa<KGEN::ParamConstantOp>(op);
    if (!isAnIndexOrPOPOrParamConstantOp && !implementsEmitsMojoInterface)
      return mlir::WalkResult::advance();

    Location opLoc = op->getLoc();
    OpBuilder builder{op};

    auto newKernel =
        wrapGenericRegionAround(builder, opLoc, op, cloneOperationBuilder);
    if (newKernel.isError()) {
      op->emitError(newKernel.takeError().get());
      return mlir::WalkResult::interrupt();
    }

    mlir::BitVector resultsToPack(op->getNumResults(), true);
    mlir::BitVector operandsToUnpack(op->getNumOperands(), true);

    if (implementsEmitsMojoInterface) {
      resultsToPack = implementsEmitsMojoInterface.resultsToPack();
      operandsToUnpack = implementsEmitsMojoInterface.operandsToUnpack();
    }

    MOGraphLoweringState::lowerMOGGABI(*newKernel, operandsToUnpack,
                                       resultsToPack);

    // If the op to lower is an index constant op, we should replace it with a
    // mgp.buffer.constant and a resource attribute. This improves compilation
    // time because we won't be hashing kernels based on the value of the
    // constant.
    if (isa<mlir::index::ConstantOp>(op)) {
      auto loweredIndexOp =
          *newKernel->getBodyBlock().getOps<mlir::index::ConstantOp>().begin();
      OpBuilder builder(loweredIndexOp);
      auto loc = loweredIndexOp.getLoc();
      uint64_t alignment =
          static_cast<uint64_t>(minimumCpuAlignment.getValue());

      int64_t indexValue = loweredIndexOp.getValue().getSExtValue();
      auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyInferAlign<int64_t>(
          {&indexValue, 1});
      auto elementsAttr =
          AlignedBytesAttr::get(ctx, blob.getDataAs<uint8_t>(), alignment);

      auto bufferConstantOp =
          MGP::BufferConstantOp::create(builder, loc, elementsAttr);
      auto dummyChain = MOGG::ChainDummyOp::create(builder, loc);
      auto newIndexOp = MGP::BufferToIndexOp::create(
          builder, loc, {dummyChain, bufferConstantOp.getResult()});
      loweredIndexOp->replaceAllUsesWith(newIndexOp);
      loweredIndexOp.erase();
      auto currentResources = newKernel->getConstantResources();
      SmallVector<Attribute> newResources(currentResources.begin(),
                                          currentResources.end());
      newResources.push_back(elementsAttr);
      newKernel->setConstantResourcesAttr(ArrayAttr::get(ctx, newResources));
    }

    op->replaceAllUsesWith(*newKernel);
    op->erase();

    return mlir::WalkResult::skip();
  };
  if (model.walk</*Order=*/mlir::WalkOrder::PreOrder>(walkerFn)
          .wasInterrupted()) {
    signalPassFailure();
  }
}

} // namespace

} // namespace M::MGP

std::unique_ptr<mlir::Pass> M::MGP::createLowerPrimitives(
    const LowerPrimitivesOptions &opts,
    std::shared_ptr<MOGG::MojoLibraryAnalysis> libraryAnalysis) {
  return std::make_unique<LowerPrimitivesPass>(opts,
                                               std::move(libraryAnalysis));
}
