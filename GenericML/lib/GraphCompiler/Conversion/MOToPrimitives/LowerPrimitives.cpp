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
#include "GenericML/MGPDialect/MGPOps.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "KGEN/CustomDialect/CustomDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
    : public MGP::impl::LowerPrimitivesBase<LowerPrimitivesPass> {
  using LowerPrimitivesBase::LowerPrimitivesBase;

  void runOnOperation() override;
};

// All POP, index and MGP Primitives that are registered we will allow fusion
// of.
static inline bool canFuseOp(Operation *op) {
  if (isa<MGP::SyncOp>(op))
    return false;

  if (isa<MGP::BufferAllocOp>(*op))
    return false;

  // Allow for custom ops starting with "testfuse." for testing purposes.
  return isa_and_nonnull<MGPDialect, KGEN::POP::POPDialect,
                         mlir::index::IndexDialect>(op->getDialect()) ||
         isa<KGEN::ParamConstantOp>(op) ||
         op->getName().getIdentifier().strref().starts_with("testfuse.");
}

static inline bool isAsync(Operation *op) {
  // Event sync should be launched in a seperate thread.
  return isa<MGP::SyncOp>(op);
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
isLowerableToMOGGCall(MLIRContext *ctx,
                      MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                      Operation *op) {
  bool isRegistered = mojoLibraryAnalysis.hasRegisteredMojoKernelFor(
      op->getName().getStringRef());
  return isRegistered;
}

// Get a dictionary of the attributes for the lowered mogg call
static DictionaryAttr getMoggCallAttrs(OpBuilder &builder, Operation *op) {
  SmallVector<NamedAttribute> attrs;

  SmallVector<Attribute> liftAttrs{};
  if (auto attrToValOp = dyn_cast<MGP::MGPLiftAttrToRTValuesOpInterface>(op)) {
    liftAttrs = attrToValOp.getLiftAttributes();
    attrs.push_back(
        builder.getNamedAttr("resources", builder.getArrayAttr(liftAttrs)));
  }

  for (auto &attr : op->getAttrs()) {
    // Attrs that need to be lifted are handled above
    if (llvm::find(liftAttrs, attr.getValue()) != liftAttrs.end())
      continue;

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
  // For anything that takes in a tensor-spec input, we need to parameterize the
  // rank so the mojo calls can use that.
  for (auto operandType : op->getOperands().getTypes()) {
    if (auto tensorSpec = dyn_cast<MGP::TensorSpecType>(operandType)) {
      attrs.push_back(builder.getNamedAttr(
          "spec_rank", builder.getIndexAttr(tensorSpec.getShape().size())));
    } else if (auto tensor = dyn_cast<MGP::DeviceTensorType>(operandType)) {
      // A scalar mgp.tensor is promoted to NDBuffer<[1]>
      size_t rank = tensor.getRank();
      attrs.push_back(
          builder.getNamedAttr("tensor_rank", builder.getIndexAttr(rank)));
      attrs.push_back(builder.getNamedAttr(
          "buffer_rank", builder.getIndexAttr(rank == 0 ? 1 : rank)));
      attrs.push_back(
          builder.getNamedAttr("type", TypeAttr::get(tensor.getElementType())));
    }
  }

  // For anything that produces a tensor output, we need to paramerized the
  // dtype.
  for (auto resultType : op->getResultTypes()) {
    if (auto tensorType = dyn_cast<MGP::DeviceTensorType>(resultType)) {
      attrs.push_back(builder.getNamedAttr(
          "type", TypeAttr::get(tensorType.getElementType())));
      attrs.push_back(builder.getNamedAttr(
          "buffer_rank", builder.getIndexAttr(tensorType.getRank() == 0
                                                  ? 1
                                                  : tensorType.getRank())));
    }
  }

  return builder.getDictionaryAttr(attrs);
}

// Transfer a operation into a mogg.call.
static ErrorOr<SmallVector<Value>>
buildMOGGCall(OpBuilder &builder, ValueRange args, Operation *op, Location loc,
              MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis) {

  KGEN::GeneratorOp kernel = nullptr;
  // Convert opaque inputs.
  SmallVector<Value> inputs(args);
  for (auto [idx, ty] : llvm::enumerate(args.getTypes())) {
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
        StringAttr::get(kernel->getContext(), "OpaquePrimitive"), std::nullopt);
    val = builder.create<MOGG::ToOpaqueFromMGPOp>(loc, moOpaqueTy, val);
    inputs[idx] = builder.create<MOGG::ToKGENOp>(
        loc, kernel.getFunctionType().getInput(idx), val,
        /*static_tensor_spec=*/nullptr, /*typeMethodDict=*/nullptr,
        isRegisterType);
  }

  // Add the call to the actual implementing function in mojo.
  MOGG::CallOp call = builder.create<MOGG::CallOp>(
      loc, op->getResultTypes(), op->getName().getStringRef(), inputs,
      ArrayAttr{}, getMoggCallAttrs(builder, op));

  SmallVector<Value> outputs(call->getResults());

  // Handle ops that may extend buffer lifetimes
  if (auto extendsLifetime = dyn_cast<GRT::GRTExtendsLifetimeInterface>(*op)) {
    for (size_t resultIdx = 0; resultIdx < op->getNumResults(); ++resultIdx)
      for (size_t operandIdx = 0; operandIdx < op->getNumOperands();
           ++operandIdx)
        if (extendsLifetime.extendsLifetime(operandIdx, resultIdx))
          outputs[resultIdx] = builder
                                   .create<MOGG::BufferBorrowOp>(
                                       loc, op->getResultTypes(),
                                       outputs[resultIdx], inputs[operandIdx])
                                   ->getResult(0);
  }

  // Convert opaque outputs
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
      return Error("missing methods for the primitive output opaque type: "
                   "__moveinit__ / __del__ not found");
    }
    bool isRegister = !kernel.getSignature().hasMemoryOnlyResult();

    // Convert opaque to MGP value.
    // TODO: Simplify mgp.opaque lowerings for MGP Primitives (GRA-812).
    Value val = outputs[idx];
    auto moOpaqueTy = MO::OpaqueType::get(
        kernel->getContext(),
        StringAttr::get(kernel->getContext(), "OpaquePrimitive"), std::nullopt);
    val = builder.create<MOGG::ToMOGGOp>(loc, moOpaqueTy, val,
                                         *methodsDictAttrOr, isRegister);
    outputs[idx] = builder.create<MOGG::ToMGPFromOpaqueOp>(
        loc, outputs[idx].getType(), val);
  }
  return outputs;
}

// Wrap a mogg.generic_compiled_region around the given operation.
static ErrorOr<MOGG::GenericCompiledRegionOp>
wrapGenericRegionAround(OpBuilder &builder, Location loc, Operation *op,
                        CompiledRegionBodyBuilder bodyBuilder) {
  auto constResources = builder.getArrayAttr({});
  if (auto attrToValOp = dyn_cast<MGP::MGPLiftAttrToRTValuesOpInterface>(op))
    constResources = builder.getArrayAttr(attrToValOp.getLiftAttributes());

  // Build compile region operation.
  SmallVector<InOutSignatureAttr::InOutSemantics> operandSemantics(
      op->getNumOperands(), InOutSignatureAttr::InOutSemantics::kNone);
  auto newKernel = builder.create<MOGG::GenericCompiledRegionOp>(
      loc, op->getResultTypes(), op->getOperands(),
      KGEN::ParamDeclArrayAttr::get(builder.getContext(), {}),
      InOutSignatureAttr::get(builder.getContext(), operandSemantics),
      builder.getBoolAttr(true),                           // isKGEN
      builder.getStringAttr(op->getName().getStringRef()), // summary
      builder.getBoolAttr(isAsync(op)),                    // isAsync
      constResources,                                      // constantResources
      builder.getBoolAttr(canFuseOp(op)),                  // canFuseIntoOther
      builder.getBoolAttr(canFuseOp(op)),                  // canBeFusedInto
      builder.getBoolAttr(canDuplicate(op))                // canDuplicate
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

  builder.create<MOGG::OutputOp>(loc, outputs.takeValue());
  return newKernel;
};

void LowerPrimitivesPass::runOnOperation() {
  ModuleOp mod = getOperation();
  MLIRContext *ctx = mod->getContext();

  auto &analysis = getAnalysis<MOGG::LoadMojoAnalysisPass>();
  ErrorOr<MOGG::MojoLibraryAnalysis *> errOr =
      analysis.initIfNeeded(ctx, kernelLibraryPaths, withDebuginfo);
  if (errOr.isError())
    return signalPassFailure();
  MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis = *errOr.takeValue();

  auto moggCallBodyBuilder =
      [&mojoLibraryAnalysis](OpBuilder &builder, Location loc, Operation *op,
                             ValueRange inputs) -> ErrorOr<SmallVector<Value>> {
    SmallVector<Value> inputVec(inputs);
    if (auto bufferConcat = dyn_cast<MGP::BufferConcatOp>(op)) {
      // Packs the variadic input buffers into tuple.
      ValueRange inBuffers =
          inputs.take_back(bufferConcat.getInBuffers().size());
      Value tuple = builder.create<MOGG::TupleCreateOp>(
          op->getLoc(), builder.getTupleType(inBuffers.getTypes()), inBuffers);

      inputVec.pop_back_n(inBuffers.size());
      inputVec.push_back(tuple);
    } else if (isa<MGP::BufferGetCachedOp, MGP::BufferConstantOp>(op)) {
      // This is the storage_ref handle to the buffer.
      Value asyncRef = builder.create<MOGG::AnyAsyncRefPlaceholderOp>(loc);
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
          builder
              .create<MOGG::BufferBorrowOp>(loc, op->getResultTypes()[0],
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

  mod.walk([&](Operation *op) {
    Location opLoc = op->getLoc();
    OpBuilder builder{op};

    // Skip any operations nested within a region.
    if (op->getParentOfType<MOGG::GenericCompiledRegionOp>() ||
        op->getParentOfType<KGEN::GeneratorOp>())
      return;

    // `arith.constant` might be created after constant folding, canonicalize
    // them into `index.constant/index.bool.constant` accordingly.
    if (auto arithConstOp = dyn_cast<mlir::arith::ConstantOp>(op)) {
      Operation *indexConstOp = nullptr;
      auto intAttr = dyn_cast<mlir::IntegerAttr>(arithConstOp.getValue());
      if (arithConstOp.getType() == builder.getIndexType()) {
        indexConstOp =
            builder.create<mlir::index::ConstantOp>(opLoc, intAttr.getValue());
      } else if (arithConstOp.getType() == builder.getI1Type()) {
        indexConstOp = builder.create<mlir::index::BoolConstantOp>(
            opLoc, builder.getBoolAttr(intAttr.getValue().getBoolValue()));
      }
      if (indexConstOp) {
        op->replaceAllUsesWith(indexConstOp);
        op->erase();
        op = indexConstOp;
        builder.setInsertionPoint(indexConstOp);
      }
      // else unhandled constant type
    }

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
        return;
      }

      op->replaceAllUsesWith(newKernel.takeValue());
      op->erase();
    }
  });
}

} // namespace

} // namespace M::MGP
