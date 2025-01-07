//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENStubBuilderHelpers.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MOToKGENLowering.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoTargetFunctions.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "Runtime/Tensor/Tensor.h"
#include "Support/AssertStream.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEAFFINEVIEWS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// TODO: Better to rename the pass, by no means a gather operation (which is
// handled in the pass) is "affine".
struct FuseAffineViewsPass
    : public impl::FuseAffineViewsBase<FuseAffineViewsPass> {
  using FuseAffineViewsBase::FuseAffineViewsBase;

  static inline bool isFusibleNonElementwise(KernelOp kernel) {
    // Cannot be elementwise kernel, that is fused differently.
    auto isElemwise = kernel.getFusionBodyLambda();
    if (isElemwise)
      return false;

    return true;
  }

  static inline bool isOutputFusibleNonElementwise(KernelOp kernel) {
    if (!isFusibleNonElementwise(kernel))
      return false;

    if (!kernel.getOutputFusionInterface().has_value())
      return false;

    // Output can only have one user.
    Value result = kernel.getResult(0);
    if (!result.hasOneUse())
      return false;
    Operation *user = *result.getUsers().begin();

    // Only fuse within the same block and not across, i.e, control flow.
    if (kernel->getBlock() != user->getBlock()) {
      return false;
    }

    return true;
  }

  static inline bool isInputFusibleNonElementwise(KernelOp kernel,
                                                  size_t inputNum,
                                                  KernelOp inputKernel) {
    if (!isFusibleNonElementwise(kernel))
      return false;

    // Only fuse within the same block and not across, i.e, control flow.
    if (kernel->getBlock() != inputKernel->getBlock())
      return false;

    return !kernel.getInputFusionInterfaceAt(inputNum).strref().empty();
  }

  void addSingletonDimTransformMOGG(Value indices,
                                    MO::TensorType outputTensorType,
                                    MO::AddSingletonDimOp addDim,
                                    OpBuilder &builder) {
    mlir::NamedAttrList attrs;
    attrs.set("rank", builder.getIndexAttr(outputTensorType.getRank()));
    attrs.set("axis", addDim.getAxis());
    attrs.set("value", builder.getIndexAttr(0));

    auto call = builder.create<MOGG::CallOp>(
        addDim.getLoc(), TypeRange{indices.getType()}, "insert_index",
        ValueRange{indices}, mlir::ArrayAttr{},
        /*kgenAttrs=*/builder.getDictionaryAttr(attrs));

    indices.replaceUsesWithIf(call->getResult(0), [&](OpOperand &operand) {
      if (operand.getOwner() == call)
        return false;
      return true;
    });
  }

  void addSplitDimMOGG(Value indices, MO::TensorType outputTensorType,
                       MO::SplitDimOp splitDim, Operation *insertPt,
                       OpBuilder &builder) {
    mlir::MLIRContext *ctx = builder.getContext();

    mlir::NamedAttrList attrs;
    attrs.set("rank", builder.getIndexAttr(outputTensorType.getRank()));
    attrs.set("axis", splitDim.getAxis());

    int64_t i = splitDim.getAxis().getValue().getLimitedValue();

    // Insert the shape calculation outside of the lambda so the index is
    // captured by the lambda rather than the pointer to the shape.
    OpBuilder shapeCalcBuilder{ctx};
    shapeCalcBuilder.setInsertionPointAfter(insertPt);

    // Get the the dimension we are splitting by. For insance for [N] -> [n1,
    // n2], this value refers to `n2`.
    Value getDim = MOGG::lowerDimToKGEN(splitDim.getLoc(), shapeCalcBuilder,
                                        splitDim.getType().getDims()[i + 1]);

    // Add the actual indices transfrom into the lambda expression.
    auto call = builder.create<MOGG::CallOp>(
        splitDim.getLoc(), TypeRange{indices.getType()}, "split_dim_indices",
        ValueRange{indices, getDim}, mlir::ArrayAttr{},
        /*kgenAttrs=*/builder.getDictionaryAttr(attrs));

    indices.replaceUsesWithIf(call->getResult(0), [&](OpOperand &operand) {
      if (operand.getOwner() == call)
        return false;
      return true;
    });
  }

  Value addGatherIndicesTranslation(Location loc, OpBuilder &builder,
                                    Value indicesAfterGather,
                                    Value indicesBuffer, Value gatherAxis,
                                    Value inputShape) {
    // The parameters to the call are bound with NDBuffer and shape, which graph
    // compiler knows how to deduce from.

    // Translate gathered indices back to the original indices.
    auto call = builder.create<MOGG::CallOp>(
        loc, TypeRange{indicesAfterGather.getType()},
        "translate_gather_indices",
        ValueRange{indicesBuffer, gatherAxis, indicesAfterGather, inputShape},
        mlir::ArrayAttr{},
        /*kgenAttrs=*/builder.getDictionaryAttr({}));

    return call.getResult(0);
  }

  // Support fusion for old-style kernels which use high level MOGG ops
  template <typename ViewOp>
  void fuseViewOpDimMOGG(MOGG::KernelOp kernel, ViewOp viewOpUser,
                         Value result) {
    MO::TensorType newType = viewOpUser.getResult().getType();

    OutputPlaceholdersOp outPh = kernel.getOutputPlaceholder();
    Value outTensor = outPh.getResults()[0];
    auto oldType = cast<MO::TensorType>(outTensor.getType());

    // Update the internal placeholder op.
    outTensor.setType(newType);

    // Insert a view to the old shape and replace everything EXCEPT the
    // output return and the internal simd store with this view.
    OpBuilder builder{outPh};
    builder.setInsertionPointAfter(outPh);

    auto shapeTy =
        MOGG::ShapeType::get(kernel.getContext(), oldType.getTypedShapeAttr());
    Value shapeInKGEN =
        MOGG::lowerShapeToKGEN(outPh->getLoc(), builder, oldType.getDims());
    Value shape = builder.create<MOGG::ToMOGGOp>(
        outPh->getLoc(), TypeRange{shapeTy}, ValueRange{shapeInKGEN});

    auto reshape = builder.create<MOGG::CallOp>(
        outPh.getLoc(), TypeRange{oldType}, "reshape_contiguous_buffer",
        ValueRange{outTensor, shape}, mlir::ArrayAttr{},
        /*kgenAttrs=*/mlir::DictionaryAttr{});
    Value reshapedOutput = reshape.getResult(0);

    outTensor.replaceUsesWithIf(reshapedOutput, [&](OpOperand &operand) {
      Operation *usr = operand.getOwner();

      // Don't replace the reshape we just added, ofc.
      if (usr == reshape)
        return false;

      // Output should be returning the same shape as the graph level
      // expects.
      if (isa<MOGG::OutputOp>(usr))
        return false;

      // The store should be operating on the tensor with the same
      // dimension as the output expects.
      if (auto callOp = dyn_cast<MOGG::CallOp>(usr)) {
        if (callOp.getMojoKernelTarget() == Utils::MOGGSimdStoreTarget &&
            callOp.getOperand(0) == outTensor) {
          return false;
        }
      }
      return true;
    });

    // 2.Transform the indices of the output lambda by fusing in the view
    // operation.
    StringAttr lambdaName = kernel.getOutputFusionInterfaceAt(0);

    kernel.walk([&](MOGG::LambdaOp lambda) {
      // Insert call to inverse of the view op to invert the indices.
      if (lambda.getSymName() == lambdaName) {
        mlir::Block &block = lambda.getBodyRegion().front();
        Value indices = block.getArgument(0);
        builder.setInsertionPoint(&block, block.begin());

        if constexpr (std::is_same_v<ViewOp, MO::AddSingletonDimOp>)
          addSingletonDimTransformMOGG(indices, oldType, viewOpUser, builder);
        else if (std::is_same_v<ViewOp, MO::SplitDimOp>)
          addSplitDimMOGG(indices, oldType, viewOpUser, reshape, builder);
      }
    });

    // Change the graph level type of the mogg kernel op and replace the
    // old view.
    result.setType(newType);
    viewOpUser.getResult().replaceAllUsesWith(result);
  }

  // Calls `reshape_contiguous_managed_tensor_slice` on the given `kgenTensor`,
  // transforming it from the equivalent `fromType` to `toType` in MO land.
  //
  // Returns the new kgen tensor value.
  static Value reshapeKGENManagedTensorSlice(
      Value kgenTensor, MO::TensorType fromType, MO::TensorType toType,
      MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis, OpBuilder &builder,
      TypeManager &typeManager, Location loc) {

    // Step 1: Find the function we want to call.
    auto reshapeKGENFunc = mojoLibraryAnalysis.getRegisteredMojoKernelFor(
        Utils::DPSManagedTensorSliceReshape);
    ASSERT_STREAM(reshapeKGENFunc, << "Could not find intrinsic "
                                   << Utils::DPSManagedTensorSliceReshape);

    ASSERT_STREAM(fromType.getDtype() == toType.getDtype(),
                  << "Expect tensors to have the same dtype");
    TypedAttr typeBinding =
        typeManager.typeMap.at(fromType.getDtype()).typeBinding;

    Value toShapeKGEN = MOGG::lowerShapeToKGEN(loc, builder, toType.getDims());
    auto toRankAttr = builder.getIndexAttr(toType.getRankNonZero());
    auto fromRankAttr = builder.getIndexAttr(std::max(1ul, fromType.getRank()));

    // construct fully specialized kgen tensor signature (managed_tensor_slice)
    SmallVector<TypedAttr> reshapeParameterValues = {typeBinding, fromRankAttr,
                                                     toRankAttr};
    auto reshapeSignature =
        reshapeKGENFunc.getSignatureGenerator().getSpecializedGenerator(
            reshapeParameterValues);

    // We may need a rebind since some complex default parameter value logic
    // may not be folded all the way at the start.
    Value toShapeKGENRebound = builder.create<KGEN::RebindOp>(
        loc, reshapeSignature.getBody().getArguments()[1], toShapeKGEN);
    auto symbol = KGEN::SymbolConstantAttr::get(
        reshapeKGENFunc, reshapeSignature, reshapeParameterValues);

    auto callArgs = SmallVector<Value>{kgenTensor, toShapeKGENRebound};
    auto reshapedTensorCall = builder.create<KGEN::CallOp>(
        reshapeKGENFunc->getLoc(), symbol, callArgs);

    return reshapedTensorCall->getResult(0);
  }

  // Returns the equivalent KGEN representation of `moType`
  static Type getKGENTensorType(MO::TensorType moType,
                                MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                                OpBuilder &builder, TypeManager &typeManager) {

    TypedAttr typeBinding =
        typeManager.typeMap.at(moType.getDtype()).typeBinding;
    uint64_t rank = std::max(1ul, moType.getRank());

    auto newRankAttr = builder.getIndexAttr(rank);
    auto toManagedTensorSlice = mojoLibraryAnalysis.getRegisteredMojoKernelFor(
        MOJO_INTRINSIC_TO_MANAGED_TENSOR_SLICE);
    ASSERT_STREAM(toManagedTensorSlice,
                  << "Could not find intrinsic "
                  << MOJO_INTRINSIC_TO_MANAGED_TENSOR_SLICE);

    // construct fully specialized kgen tensor signature (managed_tensor_slice)
    SmallVector<TypedAttr> parameterValues = {typeBinding, newRankAttr};
    auto signature =
        toManagedTensorSlice.getSignatureGenerator().getSpecializedGenerator(
            parameterValues);

    assert(signature.getBody().getNumArguments() == 2);
    assert(signature.getBody().getNumResults() == 1);
    return signature.getBody().getResults().front();
  }

  /// Given `indices` in KGEN, remap them according to provided `splitDimOp`
  static Value
  getKGENSplitDimIndices(Value kgenIndicesOperand, MO::SplitDimOp splitDimOp,
                         MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                         OpBuilder &builder) {
    // Create call to the indices mapping function
    auto splitDimIndicesFunc =
        mojoLibraryAnalysis.getRegisteredMojoKernelFor(Utils::SplitDimIndices);
    ASSERT_STREAM(splitDimIndicesFunc,
                  << "Could not find intrinsic " << Utils::SplitDimIndices);

    // construct fully specialized kgen tensor signature
    // (managed_tensor_slice)
    SmallVector<TypedAttr> splitDimIndicesParameterValues = {
        builder.getIndexAttr(splitDimOp.getInput().getType().getRankNonZero()),
        builder.getIndexAttr(splitDimOp.getAxis().getInt())};

    auto splitDimIndicesSignature =
        splitDimIndicesFunc.getSignatureGenerator().getSpecializedGenerator(
            splitDimIndicesParameterValues);

    auto symbol = KGEN::SymbolConstantAttr::get(splitDimIndicesFunc,
                                                splitDimIndicesSignature,
                                                splitDimIndicesParameterValues);

    // Calculate the split dim size for the function
    int64_t i = splitDimOp.getAxis().getValue().getLimitedValue();

    // Get the the dimension we are splitting by. For insance for [N] ->
    // [n1, n2], this value refers to `n2`.
    Value splitDim = MOGG::lowerDimToKGEN(
        splitDimOp.getLoc(), builder, splitDimOp.getType().getDims()[i + 1]);

    auto callArgs = SmallVector<Value>{kgenIndicesOperand, splitDim};

    return builder
        .create<KGEN::CallOp>(splitDimIndicesFunc->getLoc(), symbol, callArgs)
        ->getResult(0);
  }

  /// Given `indices` in KGEN, remap them according to provided `addSingleton`
  /// operation.
  static Value getKGENAddSingletonIndices(
      Value kgenIndicesOperand, MO::AddSingletonDimOp addSingleton,
      MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis, OpBuilder &builder) {
    // Create call to the indices mapping function
    auto insertIndexFunc =
        mojoLibraryAnalysis.getRegisteredMojoKernelFor(Utils::InsertIndex);
    ASSERT_STREAM(insertIndexFunc,
                  << "Could not find intrinsic " << Utils::InsertIndex);

    // construct fully specialized kgen tensor signature
    // (managed_tensor_slice)
    SmallVector<TypedAttr> insertIndexParameterValues = {
        builder.getIndexAttr(
            addSingleton.getInput().getType().getRankNonZero()),
        builder.getIndexAttr(addSingleton.getAxis().getInt()),
        builder.getIndexAttr(0)};

    auto insertIndexSignature =
        insertIndexFunc.getSignatureGenerator().getSpecializedGenerator(
            insertIndexParameterValues);

    auto symbol = KGEN::SymbolConstantAttr::get(
        insertIndexFunc, insertIndexSignature, insertIndexParameterValues);

    auto callArgs = SmallVector<Value>{kgenIndicesOperand};

    return builder
        .create<KGEN::CallOp>(insertIndexFunc->getLoc(), symbol, callArgs)
        ->getResult(0);
  }

  // Rewrites the given output lambda to make it compatible with view fusion
  // with respect to the `viewOpUser`.
  //
  // We expect the given output lambda to take in indices which correspond to
  // the input tensor type in `viewOpUser`. However, we want to rewrite the
  // indices to write to the output tensor type in `viewOpUser`. The KGEN
  // representation of this new output type is `kgenNewOutputTensor`.
  //
  // We therefore want to transform the input indices corresponding to the old
  // tensor type to the new type after the `viewOpUser`. We then take these
  // indices and rewrite the call to store the given SIMD to the new
  // `kgenNewOutputTensor` using the new indices.
  template <typename ViewOp>
  void rewriteOutputLambdaInplaceKGEN(
      KGEN::ParamDeclareRegionOp outputLambdaToRewrite,
      Value kgenNewOutputTensor, ViewOp viewOpUser, OpBuilder &builder,
      MojoLibraryAnalysis &mojoLibraryAnalysis, TypeManager &typeManager) {

    MO::TensorType newOutputTensorTy = viewOpUser.getResult().getType();

    mlir::Block &block = outputLambdaToRewrite.getBodyRegion().front();
    builder.setInsertionPoint(&block, block.begin());
    auto simdStoreFunction = mojoLibraryAnalysis.getRegisteredMojoKernelFor(
        Utils::DPSManagedTensorSliceStoreSIMD);

    // Find the op which does the write back to the tensor
    KGEN::CallOp oldStoreOp;
    block.walk([&](KGEN::CallOp op) {
      auto callSymbolName = op.getCallee().getSymbol();

      if (callSymbolName.getLeafReference() != simdStoreFunction.getSymName())
        return;

      ASSERT_STREAM(!oldStoreOp, << "Expect only one op to write to tensor");
      oldStoreOp = op;
    });

    ASSERT_STREAM(oldStoreOp, << "Could not find store operation in lambda");
    ASSERT_STREAM(oldStoreOp->getNumOperands() == 3,
                  << "Expect SIMD store operation to take 3 operands: "
                     "the tensor, indices, and simd to store");
    Value kgenIndicesOperand = oldStoreOp->getOperand(1);
    Value kgenSIMDOperand = oldStoreOp->getOperand(2);

    // get the new indices
    Value kgenNewIndicesOperand;
    if constexpr (std::is_same_v<ViewOp, MO::AddSingletonDimOp>) {
      // TODO(GEX-1482): Remove this, right now we expect this op to
      // canonicalize away to SplitDimOp.
      kgenNewIndicesOperand = getKGENAddSingletonIndices(
          kgenIndicesOperand, viewOpUser, mojoLibraryAnalysis, builder);
    } else if (std::is_same_v<ViewOp, MO::SplitDimOp>) {
      kgenNewIndicesOperand = getKGENSplitDimIndices(
          kgenIndicesOperand, viewOpUser, mojoLibraryAnalysis, builder);
    }
    ASSERT_STREAM(kgenNewIndicesOperand,
                  << "The new indices for this affine mapping were not "
                     "found! Likely using unsupported view op type: "
                  << viewOpUser);

    // Create new call to store values
    auto simdStoreInternalFunc = mojoLibraryAnalysis.getRegisteredMojoKernelFor(
        Utils::DPSManagedTensorSliceStoreSIMD);
    ASSERT_STREAM(simdStoreInternalFunc,
                  << "Could not find intrinsic "
                  << Utils::DPSManagedTensorSliceStoreSIMD);

    ASSERT_STREAM(outputLambdaToRewrite.getInputParams().size() == 1,
                  "Expect output lambda to only declare one parameter "
                  "(simd width)");
    ParamDeclAttr simdWidthAttr = outputLambdaToRewrite.getInputParams()[0];
    ASSERT_STREAM(isa<mlir::IndexType>(simdWidthAttr.getType()),
                  << "Expect output lambda to only declare one parameter "
                     "(simd width)");

    // construct fully specialized kgen tensor signature
    // (managed_tensor_slice)
    ArrayRef<TypedAttr> oldStoreOpParameters = oldStoreOp.getParamValues();

    auto strideVals = MOGG::getStridesForTensor(newOutputTensorTy, false);
    auto stridesOrError = typeManager.lowerStaticDims(builder, strideVals);
    ASSERT_STREAM(!stridesOrError, << "expected stride calculation to work!");
    SmallVector<TypedAttr> simdStoreInternalParameterValues = {
        typeManager.typeMap.at(newOutputTensorTy.getDtype())
            .typeBinding,                                  // type
        builder.getIndexAttr(newOutputTensorTy.getRank()), // rank
        KGEN::ParamDeclRefAttr::get(simdWidthAttr),        // simd_width
        oldStoreOpParameters[3],                           // alignment
        oldStoreOpParameters[4],                           // address_space
        stridesOrError.takeValue(),                        // static_strides
    };

    auto simdStoreInternalSignature =
        simdStoreInternalFunc.getSignatureGenerator().getSpecializedGenerator(
            simdStoreInternalParameterValues);

    auto symbol = KGEN::SymbolConstantAttr::get(
        simdStoreInternalFunc, simdStoreInternalSignature,
        simdStoreInternalParameterValues);

    Value kgenNewIndicesOperandRebound = builder.create<KGEN::RebindOp>(
        kgenNewIndicesOperand.getLoc(),
        simdStoreInternalSignature.getBody().getArguments()[1],
        kgenNewIndicesOperand);

    // Applying a bunch of logic on the parameters
    builder.setInsertionPointAfter(oldStoreOp);

    SmallVector<Value> callArgs = {
        kgenNewOutputTensor, kgenNewIndicesOperandRebound, kgenSIMDOperand};
    builder.create<KGEN::CallOp>(outputLambdaToRewrite->getLoc(), symbol,
                                 callArgs);

    // store op has got it handled now
    oldStoreOp.erase();
  }

  // Support fusion for new-style kernels which go directly to KGEN ops
  template <typename ViewOp>
  void fuseViewOpDimKGEN(MOGG::KernelOp kernel, ViewOp viewOpUser, Value result,
                         MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                         TypeManager &typeManager) {
    mlir::IRRewriter builder(kernel.getContext());

    // Save the old output placeholder and associated type
    OutputPlaceholdersOp oldOutputPlaceholder = kernel.getOutputPlaceholder();
    ASSERT_STREAM(oldOutputPlaceholder->getNumResults() == 1,
                  "Expect only one result for fusion!");
    auto oldOutputTensorType =
        cast<MO::TensorType>(oldOutputPlaceholder.getResults()[0].getType());

    // Create new output placeholder op which points to the new affine shape
    MO::TensorType newOutputTensorType = viewOpUser.getResult().getType();
    Type newKGENTensorType = getKGENTensorType(
        newOutputTensorType, mojoLibraryAnalysis, builder, typeManager);
    SmallVector<Type> newOutputPlaceHolderTy = {newOutputTensorType};
    builder.setInsertionPoint(oldOutputPlaceholder);
    OutputPlaceholdersOp newOutputPlaceholder =
        builder.create<OutputPlaceholdersOp>(oldOutputPlaceholder->getLoc(),
                                             newOutputPlaceHolderTy);

    // Create new output tensor spec
    // NOTE: this does not contain the output lambda function. If we expect
    // to need this, we can add it later, but this would also require relaxing
    // the expected type of the signature (as now we might have lambdas whose
    // index list length does not match rank).
    ErrorOr<KGEN::ParamDeclAttr> maybeNewOutputTensorSpec =
        M::MO::bindStaticTensorSpecAttrWithoutLambdas(
            kernel, newOutputPlaceholder->getResult(0), mojoLibraryAnalysis,
            typeManager, builder, "affine_view_fusion_tensor_spec");
    ASSERT_STREAM(!maybeNewOutputTensorSpec.isError(),
                  << "Could not make spec "
                  << maybeNewOutputTensorSpec.takeError());
    KGEN::ParamDeclAttr newOutputTensorSpec =
        maybeNewOutputTensorSpec.takeValue();

    // Create proper lowering to kgen with annotation
    auto newResultToKGEN = builder.create<MOGG::ToKGENOp>(
        newOutputPlaceholder->getLoc(), newKGENTensorType,
        newOutputPlaceholder->getResult(0));
    newResultToKGEN.setStaticTensorSpecNameAttr(newOutputTensorSpec.getName());

    // Replace the kernel output with the new tensor output
    kernel.getOutputOp()->setOperands(
        ValueRange{newOutputPlaceholder->getResult(0)});

    // Get the old tensor representation, and replace its use with a version of
    // the new output tensor reshaped into the old tensor shape.
    Value kgenOldTensor = reshapeKGENManagedTensorSlice(
        newResultToKGEN, newOutputTensorType, oldOutputTensorType,
        mojoLibraryAnalysis, builder, typeManager, newResultToKGEN->getLoc());
    for (Operation *user :
         llvm::make_early_inc_range(oldOutputPlaceholder->getUsers())) {
      auto asOp = dyn_cast<MOGG::ToKGENOp>(user);
      ASSERT_STREAM(asOp, << "Expected only ToKGEN uses.");
      asOp.getResult().replaceAllUsesWith(kgenOldTensor);
      asOp->erase();
    }

    // We can now safely erase the old output placeholder!
    oldOutputPlaceholder->erase();

    // Find the output lambda region. We will rewrite this to take into account
    // the affine transformation.
    StringAttr outputLambdaToRewriteName = kernel.getOutputFusionInterfaceAt(0);
    ASSERT_STREAM(kernel.getOutputFusionInterface()->size() == 1,
                  "Expect only one output fusion lambda");
    KGEN::ParamDeclareRegionOp outputLambdaToRewrite;
    kernel.walk([&](KGEN::ParamDeclareRegionOp regionDeclareOp) {
      // Insert call to inverse of the view op to invert the indices.
      if (regionDeclareOp.getParamDecl().getName() ==
          outputLambdaToRewriteName) {
        outputLambdaToRewrite = regionDeclareOp;
        return WalkResult::interrupt();
      }

      // We only care about the top level regions so can do skip.
      return WalkResult::skip();
    });
    ASSERT_STREAM(outputLambdaToRewrite,
                  << "Expected to find the region which defines the output "
                     "lambda for the kernel!");

    // Finally rewrite the output lambda.
    rewriteOutputLambdaInplaceKGEN(outputLambdaToRewrite,
                                   newResultToKGEN.getResult(), viewOpUser,
                                   builder, mojoLibraryAnalysis, typeManager);

    // Propagate the fact that we must call the output lambda now it exists.
    kernel.setKGENKernelParamValueIfPresent(
        mojoLibraryAnalysis, KGEN::MOGGPreElab::kMOGGLambdasHaveFusionLabel,
        BoolAttr::get(&getContext(), true));

    // Change graph level type of the mogg kernel op and replace the
    // old view.
    result.setType(newOutputTensorType);
    viewOpUser.getResult().replaceAllUsesWith(result);
  }

  template <typename ViewOp>
  void fuseViewOpDim(MOGG::KernelOp kernel, ViewOp viewOp, Value result,
                     MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                     TypeManager &typeManager) {
    if (kernel.getIsKGEN())
      fuseViewOpDimKGEN(kernel, viewOp, result, mojoLibraryAnalysis,
                        typeManager);
    else
      fuseViewOpDimMOGG(kernel, viewOp, result);

    // construct the new summary string
    SmallVector<Operation *> ops{kernel, viewOp};
    kernel.setSummary(getSummaryString(ops, FusionType::HARD));
  }

  void fuseGatherInputKernel(KernelOp kernel, KernelOp gatherOp,
                             unsigned toFuseIdx) {
    OpBuilder builder(kernel->getContext());

    SmallVector<Value> gatherOperands = gatherOp.getOperands();
    MO::GatherOpAdaptor adaptor(gatherOperands);

    if (auto inputTp = cast<MO::TensorType>(adaptor.getInput().getType());
        inputTp.getDeviceRef() &&
        inputTp.getDeviceRef().getLabel() != kCPULabel) {
      builder.setInsertionPoint(kernel);
      Value deviceAxis = builder.create<MO::TransferOp>(
          gatherOp->getLoc(),
          cast<MO::TensorType>(adaptor.getAxis().getType())
              .cloneWith(inputTp.getDeviceRef()),
          adaptor.getAxis());
      gatherOperands[/*AxisOpNum=*/2] = deviceAxis;
    }

    // Update the input.
    kernel->eraseOperand(toFuseIdx);
    kernel->insertOperands(toFuseIdx, gatherOperands);

    StringAttr lambdaName = kernel.getInputFusionInterfaceAt(toFuseIdx);
    ArrayRef interfaces = kernel.getInputFusionInterface()->getValue();
    ArrayRef unfusedHead = interfaces.take_front(toFuseIdx);
    SmallVector<Attribute> fusedInterfaces(unfusedHead.begin(),
                                           unfusedHead.end());

    // Update the block argument.
    Value unFusedInput = kernel.getBody().getArgument(toFuseIdx);
    for (auto [opNum, val] : llvm::enumerate(gatherOperands)) {
      // Insert right after the current value being fused to avoid messing up
      // the loop iteration, in which the function is called.
      unsigned idx = toFuseIdx + 1 + opNum;
      Value fusedVal =
          kernel.getBody().insertArgument(idx, val.getType(), val.getLoc());
      // If this is the input, replace the use to the fused input.
      if (val == adaptor.getInput()) {
        unFusedInput.replaceAllUsesWith(fusedVal);
        fusedInterfaces.emplace_back(lambdaName);
      } else {
        fusedInterfaces.emplace_back(builder.getStringAttr(""));
      }
    }
    kernel.getBody().eraseArgument(toFuseIdx);
    ArrayRef unfusedTail = interfaces.drop_front(toFuseIdx + 1);
    fusedInterfaces.append(unfusedTail.begin(), unfusedTail.end());

    // Generate fused body.
    builder.setInsertionPointToStart(&kernel.getBody().front());
    MO::GatherOpAdaptor bodyAdaptor(kernel.getBody().getArguments().slice(
        toFuseIdx, gatherOp->getNumOperands()));

    // 1st, Extract the shape from the fused input.
    Value fusedInput = bodyAdaptor.getInput();
    auto resultTy = cast<MO::TensorType>(fusedInput.getType());
    auto shape = MOGG::ShapeType::get(builder.getContext(),
                                      resultTy.getTypedShapeAttr());

    Value shapeInKGEN =
        MOGG::lowerShapeToKGEN(gatherOp->getLoc(), builder, resultTy.getDims());
    Value fusedInputShape = builder.create<MOGG::ToMOGGOp>(
        gatherOp->getLoc(), TypeRange{shape}, ValueRange{shapeInKGEN});

    // 2st, Transform the input lambda body.
    kernel.walk([&](MOGG::LambdaOp lambda) {
      // Insert call to inverse of the view op to invert the indices.
      if (lambda.getSymName() == lambdaName) {
        mlir::Block &block = lambda.getBodyRegion().front();
        Value indicesAfterGather = block.getArgument(0);
        builder.setInsertionPoint(&block, block.begin());

        auto indicesBeforeGather = addGatherIndicesTranslation(
            gatherOp->getLoc(), builder, indicesAfterGather,
            bodyAdaptor.getIndices(), bodyAdaptor.getAxis(), fusedInputShape);

        indicesAfterGather.replaceAllUsesExcept(
            indicesBeforeGather, indicesBeforeGather.getDefiningOp());
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    kernel.setInputFusionInterfaceAttr(builder.getArrayAttr(fusedInterfaces));
    kernel.setSummaryAttr(
        getSummaryString({gatherOp, kernel}, FusionType::HARD));
    // Trivially dead.
    if (gatherOp->getUsers().empty())
      gatherOp->erase();
  }

  void handleGatherFusion(MOGG::KernelOp kernel) {
    // Handles fusing *operands* into the given `kernel` if they arise from
    // a gather op.
    for (int idx = kernel.getNumOperands() - 1; idx >= 0; idx--) {
      OpOperand &operand = kernel->getOpOperand(idx);
      if (auto gatherKernel = operand.get().getDefiningOp<MOGG::KernelOp>()) {
        // Pattern match based on summary.
        if (!gatherKernel.getSummary().has_value())
          continue;

        StringRef baseKernelName = gatherKernel.getSummary()->rsplit(':').first;
        if (!gatherKernel.getSummary().has_value() ||
            baseKernelName != MO::GatherOp::getOperationName() ||
            !isInputFusibleNonElementwise(kernel, operand.getOperandNumber(),
                                          gatherKernel))
          return;

        M::IntArrayElementsAttr attr;

        MO::GatherOpAdaptor adaptor(gatherKernel.getOperands());
        if (mlir::matchPattern(adaptor.getAxis(), mlir::m_Constant(&attr))) {
          int64_t axis = (*attr.begin()).getSExtValue();
          unsigned inputRank =
              cast<MO::TensorType>(adaptor.getInput().getType()).getRank();
          // Try doing input fusion on gather only when we can statically
          // verify that the gather axis is not the last dim, otherwise, it
          // might break the `simd_load`.
          if (axis != -1 && axis != inputRank - 1)
            fuseGatherInputKernel(kernel, gatherKernel,
                                  operand.getOperandNumber());
        }
      }
    }
  }

  void runOnOperation() override {
    auto graph = getOperation();

    // Initialize mojoLibraryAnalysis
    auto &analysis = getAnalysis<MOGG::LoadMojoAnalysisPass>();
    ErrorOr<MOGG::MojoLibraryAnalysis *> errOr =
        analysis.initIfNeeded(&getContext(), {});
    if (errOr.isError()) {
      emitError(getOperation().getLoc(),
                Twine("[MO_TO_MOGG] Failed to load MOJO analysis: ") +
                    errOr.getError());
      return;
    }
    MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis = *errOr.takeValue();

    // Initialize type manager
    ErrorOr<TypeManager> maybeTypeManager = MOGG::TypeManager::get(
        mojoLibraryAnalysis.getContext(), mojoLibraryAnalysis);
    if (maybeTypeManager.isError()) {
      emitError(getOperation().getLoc(), maybeTypeManager.getError());
      return;
    }
    TypeManager typeManager = maybeTypeManager.get();

    graph.walk([&](MOGG::KernelOp kernel) {
      // These fusions fuse `kernel` into users of the kernel for certain ops.
      if (isOutputFusibleNonElementwise(kernel)) {
        Value result = kernel.getResult(0);
        Operation *user = *result.getUsers().begin();
        if (auto addDim = dyn_cast_or_null<MO::AddSingletonDimOp>(user))
          fuseViewOpDim(kernel, addDim, result, mojoLibraryAnalysis,
                        typeManager);
        else if (auto split = dyn_cast_or_null<MO::SplitDimOp>(user))
          fuseViewOpDim(kernel, split, result, mojoLibraryAnalysis,
                        typeManager);
      }

      // These fusions fuse operands into the kernel.
      // TODO(GEX-1506): Support gather fusion for new API.
      if (!kernel.getIsKGEN())
        handleGatherFusion(kernel);

      MOGG::addForwardDeclsOfReferencedFuncs(
          kernel, kernel->getParentOfType<ModuleOp>(), mojoLibraryAnalysis);
    });
  }
};

} // namespace
} // namespace M::MOGG
