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
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GraphRT/GRTDialect/GRTOps.h"
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

using CompiledRegionBodyBuilder =
    llvm::function_ref<ErrorOr<SmallVector<Value>>(OpBuilder &, Location,
                                                   Operation *, ValueRange)>;
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

// Anything we find a registry for, lower to generic regions
static inline bool
isLowerableToMOGGCall(MLIRContext *ctx,
                      MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                      Operation *op) {
  // Operations that involve runtime slots should not be lowered to mojo
  // because they need special slot offsetting during inlining
  if (isa<MGP::BufferCacheOp, MGP::RuntimeCreateOp, MGP::BufferRemoveCachedOp,
          MGP::ModuleLoadOp, MGP::ExecuteGenericOp>(op))
    return false;

  // Constant external ops should not be lowered to mojo so they can be
  // updated with a prefix during inlining.
  if (isa<MGP::BufferConstantExternalOp>(op))
    return false;

  bool isRegistered = mojoLibraryAnalysis.hasRegisteredMojoKernelFor(
      op->getName().getStringRef());
  return isRegistered;
}

// Get a dictionary of the attributes for the lowered mogg call
static DictionaryAttr getMoggCallAttrs(OpBuilder &builder, Operation *op) {
  SmallVector<NamedAttribute> attrs;

  SmallVector<NamedAttribute> needToParameterize{};
  if (auto attrToValOp = dyn_cast<MGP::MGPSelectiveLowerAttrOpInterface>(op)) {
    SmallVector<Attribute> liftAttrs = attrToValOp.getLiftableAttributes();
    if (!liftAttrs.empty())
      attrs.push_back(
          builder.getNamedAttr("resources", builder.getArrayAttr(liftAttrs)));

    needToParameterize = attrToValOp.getParameterizableAttributes();
  } else {
    needToParameterize.assign(op->getAttrs().begin(), op->getAttrs().end());
  }

  for (auto &attr : needToParameterize) {
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
    } else if (auto cmpPredAttr = dyn_cast<mlir::index::IndexCmpPredicateAttr>(
                   attr.getValue())) {
      // For IndexCmpPredicate, we stringify it and passed as a StringLiteral.
      StringRef predStr =
          mlir::index::stringifyIndexCmpPredicate(cmpPredAttr.getValue());
      attrs.push_back(
          builder.getNamedAttr("pred", builder.getStringAttr(predStr)));
    } else {
      attrs.push_back(attr);
    }
  }

  TypeAttr dtype = nullptr;
  auto checkOrAssignDType = [&](TypeAttr toAssign) {
    if (!dtype)
      dtype = toAssign;
    ASSERT_STREAM(dtype == toAssign, "Conflict DType founded");
  };

  // For anything that takes in a tensor-spec input, we need to parameterize the
  // rank so the mojo calls can use that.
  for (auto operandType : op->getOperands().getTypes()) {
    if (auto tensorSpec = dyn_cast<MGP::TensorSpecType>(operandType)) {
      attrs.push_back(builder.getNamedAttr(
          "spec_rank", builder.getIndexAttr(tensorSpec.getShape().size())));

      checkOrAssignDType(TypeAttr::get(tensorSpec.getElementType()));
    } else if (auto tensor = dyn_cast<MGP::DeviceTensorType>(operandType)) {
      // A scalar mgp.tensor is promoted to NDBuffer<[1]>
      size_t rank = tensor.getRank();
      attrs.push_back(
          builder.getNamedAttr("tensor_rank", builder.getIndexAttr(rank)));
      attrs.push_back(builder.getNamedAttr(
          "buffer_rank", builder.getIndexAttr(rank == 0 ? 1 : rank)));

      checkOrAssignDType(TypeAttr::get(tensor.getElementType()));
    }
  }

  // For anything that produces a tensor output, we need to paramerized the
  // dtype.
  for (auto resultType : op->getResultTypes()) {
    if (auto tensorType = dyn_cast<MGP::DeviceTensorType>(resultType)) {
      checkOrAssignDType(TypeAttr::get(tensorType.getElementType()));
      attrs.push_back(builder.getNamedAttr(
          "buffer_rank", builder.getIndexAttr(tensorType.getRank() == 0
                                                  ? 1
                                                  : tensorType.getRank())));
    }
  }

  if (dtype)
    attrs.push_back(builder.getNamedAttr("dtype", dtype));

  return builder.getDictionaryAttr(attrs);
}

// Transfer a operation into a mogg.call.
static ErrorOr<SmallVector<Value>>
buildMOGGCall(OpBuilder &builder, ValueRange args, Operation *op, Location loc,
              MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis) {

  KGEN::GeneratorOp kernel = nullptr;

  // Remove chains from the inputs as they lower to NOPs.
  SmallVector<Value> inputs;
  for (auto [idx, val] : llvm::enumerate(args)) {
    if (!isa<MEF::ChainType>(val.getType()))
      inputs.push_back(val);
  }

  // Filter out returned chain type.
  SmallVector<Type> callResultTps = llvm::filter_to_vector(
      op->getResultTypes(), [](Type t) { return !isa<MEF::ChainType>(t); });

  // Add the call to the actual implementing function in mojo.
  MOGG::CallOp call = MOGG::CallOp::create(
      builder, loc, callResultTps, op->getName().getStringRef(), inputs,
      ArrayAttr{}, getMoggCallAttrs(builder, op));

  SmallVector<Value> outputs;
  unsigned resultIdx = 0;
  for (Type t : op->getResultTypes()) {
    if (isa<MEF::ChainType>(t)) {
      // We create a separate op for chain creation, this allows us to optimize
      // redundant chains independently from the MOGG.call itself.
      outputs.push_back(MOGG::CreateReadyChainOp::create(
          builder, loc, MEF::ChainType::get(builder.getContext())));
    } else {
      outputs.push_back(call->getResult(resultIdx++));
    }
  }

  // Handle ops that may extend buffer lifetimes
  if (auto extendsLifetime = dyn_cast<GRT::GRTExtendsLifetimeInterface>(*op)) {
    for (size_t resultIdx = 0; resultIdx < op->getNumResults(); ++resultIdx) {

      size_t numChains = 0;
      for (size_t operandIdx = 0; operandIdx < op->getNumOperands();
           ++operandIdx) {

        // Skip chains.
        if (isa<MEF::ChainType>(op->getOperandTypes()[operandIdx])) {
          numChains++;
          continue;
        }

        if (extendsLifetime.extendsLifetime(operandIdx, resultIdx)) {
          outputs[resultIdx] =
              MOGG::BufferBorrowOp::create(builder, loc, op->getResultTypes(),
                                           outputs[resultIdx],
                                           inputs[operandIdx - numChains])
                  ->getResult(0);
        }
      }
    }
  }

  // Convert opaque outputs
  for (auto [idx, ty] : llvm::enumerate(op->getResultTypes())) {
    if (!isa<MGP::OpaqueType>(ty))
      continue;

    // Need to add extra ops for opaque outputs.
    // Load the type infos from the kernel.
    if (!kernel) {
      kernel = mojoLibraryAnalysis.getRegisteredMojoKernelFor(
          op->getName().getStringRef());
    }
    ASSERT_STREAM(kernel != nullptr, << "kernel `"
                                     << op->getName().getStringRef()
                                     << "` not found");

    FailureOr<DictionaryAttr> methodsDictAttrOr =
        MOGG::buildMojoObjectInfosOutputMethodsDict(
            kernel, ArrayRef<StringRef>{"__moveinit__", "__del__"});
    if (failed(methodsDictAttrOr)) {
      return Error("missing methods for the primitive output opaque type: "
                   "__moveinit__ / __del__ not found");
    }
    bool isRegister =
        !kernel.getFuncTypeGenerator().getBody().hasMemoryOnlyResult();

    // Convert opaque to MGP value.
    // TODO: Simplify mgp.opaque lowerings for MGP Primitives (GEX-812).
    Value val = outputs[idx];
    auto moOpaqueTy = MO::OpaqueType::get(
        StringAttr::get(kernel->getContext(), "OpaquePrimitive"));
    val = MOGG::ToMOGGOp::create(builder, loc, moOpaqueTy, val,
                                 *methodsDictAttrOr, isRegister);
    outputs[idx] = MOGG::ToMGPFromOpaqueOp::create(builder, loc,
                                                   outputs[idx].getType(), val);
  }
  return outputs;
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
};

void LowerPrimitivesPass::runOnOperation() {
  ModelOp model = getOperation();
  MLIRContext *ctx = model->getContext();

  auto analysisOr = getLibraryAnalysis();
  if (analysisOr.isError()) {
    emitError(getOperation().getLoc(),
              llvm::formatv("[{0}] Failed to load Mojo analysis: {1}",
                            getPassName(), analysisOr.getError()));
    return signalPassFailure();
  }
  MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis = *analysisOr.takeValue();

  auto moggCallBodyBuilder =
      [&mojoLibraryAnalysis](OpBuilder &builder, Location loc, Operation *op,
                             ValueRange inputs) -> ErrorOr<SmallVector<Value>> {
    SmallVector<Value> inputVec(inputs);
    if (auto bufferConcat = dyn_cast<MGP::BufferConcatOp>(op)) {

      // The inputs will be in the form [chain, outBuffer, *inBuffers] so we
      // skip over the first two to get the input buffers.
      SmallVector<Value> ins;
      for (size_t i = 0; i < bufferConcat.getInBuffers().size(); ++i)
        ins.push_back(inputs[i + 2]);

      // Packs the variadic input buffers into tuple.
      Value tuple = MOGG::AsTupleOp::create(
          builder, op->getLoc(),
          builder.getTupleType(bufferConcat.getInBuffers().getTypes()), ins);

      // Rebuild the input vector.
      inputVec.clear();
      // Chain is dropped then output is the new first argument.
      inputVec.push_back(inputs[1]);
      inputVec.push_back(tuple);
      // Device context is at the end.
      inputVec.push_back(inputs[inputs.size() - 1]);
    } else if (isa<MGP::BufferGetCachedOp, MGP::BufferConstantOp>(op)) {
      // This is the storage_ref handle to the buffer.
      Value asyncRef = MOGG::AnyAsyncRefPlaceholderOp::create(builder, loc);
      // Both buffer get cache/constant create a new storage_ref handle.
      // Differently only buffer.get_cache inherit the lifetime, and constant
      // buffer only require a nullptr handle (to mark that this is a
      // non-tracking memory).
      if (isa<MGP::BufferGetCachedOp>(op))
        inputVec.push_back(asyncRef);
      ErrorOr<SmallVector<Value>> errOr =
          buildMOGGCall(builder, inputVec, op, loc, mojoLibraryAnalysis);
      if (errOr)
        return errOr;

      // Instead of creating a new AsyncValue for the buffer, we extend the
      // lifetime of a cached buffer.
      SmallVector<Value> outputs = errOr.takeValue();
      ValueRange ret =
          MOGG::BufferBorrowOp::create(builder, loc, op->getResultTypes()[0],
                                       outputs.front(), asyncRef)
              ->getResults();

      // Override the result.
      outputs.assign(ret.begin(), ret.end());
      return outputs;
    }
    return buildMOGGCall(builder, inputVec, op, loc, mojoLibraryAnalysis);
  };

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

/// TODO(GEX-2486): Implement them one-by-one and remove them from the list each
/// time (the TEMPORARY defines).
#define MGP_PRIMITIVES_TEMPORARY_EXCLUSION_LIST                                \
  MGP::BufferAllocOp, MGP::BufferConcatOp, MGP::BufferConstantOp,              \
      MGP::BufferConstantExternalOp, MGP::BufferDeviceToDeviceOp,              \
      MGP::BufferDeviceToHostOp, MGP::BufferGetSizeOp,                         \
      MGP::BufferHostToDeviceOp, MGP::BufferRemoveCachedOp,                    \
      MGP::BufferSetWithIndexOp, MGP::BufferSliceOp, MGP::BufferToBoolOp,      \
      MGP::BufferToIndexOp, MGP::SyncOp, MGP::TensorCreateOp,                  \
      MGP::TensorExtractBufferOp, MGP::TensorExtractTensorSpecOp,              \
      MGP::TensorSpecCreateOp
#define DIALECT_TEMPORARY_EXCLUSION_LIST                                       \
  mlir::index::IndexDialect, KGEN::POP::POPDialect

  auto isPartOfAnotherDialect = [](Operation *op) {
    if (isa_and_nonnull<mlir::index::IndexDialect, KGEN::POP::POPDialect>(
            op->getDialect())) {
      return true;
    }

    // Unsure if this appears in the IR today.
    return isa<KGEN::ParamConstantOp>(op);
  };
  auto isPartOfATemporarilyExcludedDialect = [](Operation *op) {
    if (isa_and_nonnull<DIALECT_TEMPORARY_EXCLUSION_LIST>(op->getDialect()))
      return true;

    return isa<KGEN::ParamConstantOp>(op);
  };

  if (MOGG::useExperimentalMojoPath()) {
    const auto walkerFn = [&](Operation *op) {
      auto implementsEmitsMojoInterface =
          dyn_cast<MOGG::EmitsMojoInterface>(op);

      // Skip any operations nested within a region.
      if (isa<MOGG::GenericCompiledRegionOp, MOGG::KernelStubOp>(op))
        return mlir::WalkResult::skip();

      // Constant external ops should not be lowered to mojo so they can be
      // updated with a prefix during inlining.
      if (isa<MGP::BufferConstantExternalOp>(op))
        return mlir::WalkResult::advance();

      if (isPartOfAnotherDialect(op)) {
        if (isPartOfATemporarilyExcludedDialect(op))
          return mlir::WalkResult::advance();
      } else {
        /// TODO(GEX-2588): Remove this. These are not part of the exclusion
        /// list because they are not lowered in the KGEN codepath.
        if (isa<MGP::BufferCacheOp, MGP::RuntimeCreateOp,
                MGP::BufferRemoveCachedOp, MGP::ModuleLoadOp,
                MGP::ExecuteGenericOp>(op)) {
          return mlir::WalkResult::advance();
        }

        if (isa<MGP_PRIMITIVES_TEMPORARY_EXCLUSION_LIST>(op)) {
          ASSERT_STREAM(
              !implementsEmitsMojoInterface,
              << op->getName()
              << " emits Mojo but it was not removed from the exclusion list");
          return mlir::WalkResult::advance();
        }

        if (!mojoLibraryAnalysis.hasRegisteredMojoKernelFor(
                op->getName().getStringRef())) {
          // Operations that do not have a registration function can't be
          // lowered to a kernel.
          return mlir::WalkResult::advance();
        }

        ASSERT_STREAM(implementsEmitsMojoInterface,
                      << op->getName()
                      << " has a Mojo implementation but it does not implement "
                         "the EmitsMojoInterface");
      }
      Location opLoc = op->getLoc();
      OpBuilder builder{op};

      auto newKernel =
          wrapGenericRegionAround(builder, opLoc, op, cloneOperationBuilder);
      if (newKernel.isError()) {
        op->emitError(newKernel.takeError().get());
        return mlir::WalkResult::interrupt();
      }
      newKernel->setEmitsMojo(true);

      /// TODO(GEX-2588): Some of the operations covered by this issue will need
      /// their inputs or outputs in the async format (ex: mgp.buffer.cache or
      /// mgp.buffer.get_cached).
      MOGraphLoweringState::lowerMOGGABI(
          *newKernel, implementsEmitsMojoInterface.operandsToUnpack(),
          implementsEmitsMojoInterface.resultsToPack(),
          /*callsThrowingMojoStub=*/false);

      op->replaceAllUsesWith(*newKernel);
      op->erase();

      return mlir::WalkResult::skip();
    };
    if (model.walk</*Order=*/mlir::WalkOrder::PreOrder>(walkerFn)
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }
  }

  model.walk</*Order=*/mlir::WalkOrder::PreOrder>([&](Operation *op) {
    Location opLoc = op->getLoc();
    OpBuilder builder{op};

    // Skip any operations nested within a region.
    if (isa<MOGG::GenericCompiledRegionOp, MOGG::KernelStubOp>(op))
      return mlir::WalkResult::skip();

    CompiledRegionBodyBuilder bodyBuilder(nullptr);
    if ((isa_and_nonnull<mlir::index::IndexDialect>(op->getDialect()) ||
         isa_and_nonnull<KGEN::POP::POPDialect>(op->getDialect()) ||
         isa<KGEN::ParamConstantOp>(op))) {
      // Clone index operations and pop operations when necessary.
      bodyBuilder = cloneOperationBuilder;
    } else if (isLowerableToMOGGCall(ctx, mojoLibraryAnalysis, op)) {
      bodyBuilder = moggCallBodyBuilder;
    }

    if (bodyBuilder) {
      ErrorOr<MOGG::GenericCompiledRegionOp> newKernel =
          wrapGenericRegionAround(builder, opLoc, op, bodyBuilder);

      if (newKernel.isError()) {
        op->emitError(newKernel.takeError().get());
        signalPassFailure();
        return mlir::WalkResult::interrupt();
      }

      op->replaceAllUsesWith(newKernel.takeValue());
      op->erase();
      return mlir::WalkResult::skip();
    }

    return mlir::WalkResult::advance();
  });
}

} // namespace

} // namespace M::MGP

std::unique_ptr<mlir::Pass> M::MGP::createLowerPrimitives(
    const LowerPrimitivesOptions &opts,
    std::shared_ptr<MOGG::MojoLibraryAnalysis> libraryAnalysis) {
  return std::make_unique<LowerPrimitivesPass>(opts,
                                               std::move(libraryAnalysis));
}
