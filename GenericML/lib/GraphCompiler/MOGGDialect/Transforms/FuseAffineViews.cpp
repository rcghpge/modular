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
#include "KGEN/MOGGPreElab/MOGGDecorators.h"
#include "Runtime/Tensor/Tensor.h"
#include "Support/AssertStream.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/CSE.h"
#include <KGEN/MOGGPreElab/MOGGUtils.h>

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

  // Calls `reshape_contiguous_managed_tensor_slice` on the given `kgenTensor`,
  // transforming it from the equivalent `fromType` to `toType` in MO land.
  //
  // Returns the new kgen tensor value.
  static Value reshapeKGENManagedTensorSlice(
      MOGG::ToKGENOp kgenTensor, MO::TensorType fromType, MO::TensorType toType,
      MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis, OpBuilder &builder,
      TypeManager &typeManager, Location loc) {

    ASSERT_STREAM(fromType.getDtype() == toType.getDtype(),
                  << "Expect tensors to have the same dtype");
    TypedAttr typeBinding = typeManager.lookupTypeBinding(fromType.getDtype());

    Value toShapeKGEN = MOGG::lowerShapeToKGEN(loc, builder, toType.getDims());
    auto toRankAttr = builder.getIndexAttr(toType.getRankNonZero());
    auto fromRankAttr = builder.getIndexAttr(std::max(1ul, fromType.getRank()));

    // construct fully specialized kgen tensor signature (managed_tensor_slice)
    SmallVector<TypedAttr> reshapeParameterValues = {typeBinding, fromRankAttr,
                                                     toRankAttr};

    ErrorOr<KGEN::SymbolConstantAttr> reshapeKGENSymbol =
        mojoLibraryAnalysis.getSpecializedKernel(
            Utils::DPSManagedTensorSliceReshape, reshapeParameterValues);
    ASSERT_STREAM(mlir::succeeded(reshapeKGENSymbol),
                  << reshapeKGENSymbol.takeError());

    FuncType reshapeSignature = reshapeKGENSymbol->getType().getBody();

    // We may need a rebind since some complex default parameter value logic
    // may not be folded all the way at the start.
    Value toShapeKGENRebound = builder.create<KGEN::RebindOp>(
        loc, reshapeSignature.getArguments()[1], toShapeKGEN);

    auto callArgs = SmallVector<Value>{kgenTensor, toShapeKGENRebound};
    auto reshapedTensorCall =
        builder.create<KGEN::CallOp>(loc, *reshapeKGENSymbol, callArgs);

    return reshapedTensorCall->getResult(0);
  }

  // Returns the equivalent KGEN representation of `moType`
  static Type getKGENTensorType(MO::TensorType moType,
                                MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                                OpBuilder &builder, TypeManager &typeManager) {

    TypedAttr typeBinding = typeManager.lookupTypeBinding(moType.getDtype());
    uint64_t rank = std::max(1ul, moType.getRank());

    auto newRankAttr = builder.getIndexAttr(rank);

    // construct fully specialized kgen tensor signature (managed_tensor_slice)
    SmallVector<TypedAttr> parameterValues = {typeBinding, newRankAttr};

    auto toManagedTensorSlice = mojoLibraryAnalysis.getSpecializedKernel(
        MOJO_INTRINSIC_TO_MANAGED_TENSOR_SLICE, parameterValues);
    ASSERT_STREAM(succeeded(toManagedTensorSlice),
                  << toManagedTensorSlice.takeError());

    FuncType signature = toManagedTensorSlice->getType().getBody();

    assert(signature.getNumArguments() == 2);
    assert(signature.getNumResults() == 1);
    return signature.getResults().front();
  }

  /// Given `indices` in KGEN, remap them according to provided `splitDimOp`
  template <typename SplitOrMergeDimOp>
  static Value getKGENSplitOrMergeDimIndices(
      Value kgenIndicesOperand, SplitOrMergeDimOp splitOrMergeDimOp,
      MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis, OpBuilder &builder) {
    // construct fully specialized kgen tensor signature
    // (managed_tensor_slice)
    SmallVector<TypedAttr> splitOrMergeDimIndicesParameterValues = {
        builder.getIndexAttr(
            splitOrMergeDimOp.getInput().getType().getRankNonZero()),
        builder.getIndexAttr(splitOrMergeDimOp.getAxis().getInt())};

    std::optional<llvm::StringLiteral> funcName;
    if constexpr (std::is_same_v<SplitOrMergeDimOp, MO::SplitDimOp>) {
      funcName = Utils::SplitDimIndices;
    } else if constexpr (std::is_same_v<SplitOrMergeDimOp, MO::MergeDimOp>) {
      funcName = Utils::MergeDimIndices;
    }
    ASSERT_STREAM(funcName, "funcName should be non-null");

    // Create call to the indices mapping function
    auto splitOrMergeDimIndicesFunc = mojoLibraryAnalysis.getSpecializedKernel(
        *funcName, splitOrMergeDimIndicesParameterValues);
    ASSERT_STREAM(succeeded(splitOrMergeDimIndicesFunc),
                  << splitOrMergeDimIndicesFunc.takeError());

    // Calculate the split dim size for the function
    int64_t i = splitOrMergeDimOp.getAxis().getValue().getLimitedValue();

    // Get the dimension we are splitting / merging by.
    // For split ([N] -> [n1, n2]) or merge ([n1, n2] -> [N]), this value
    // refers to `n2`.
    TypedAttr axis;
    if constexpr (std::is_same_v<SplitOrMergeDimOp, MO::SplitDimOp>) {
      axis = splitOrMergeDimOp.getType().getDims()[i + 1];
    } else if constexpr (std::is_same_v<SplitOrMergeDimOp, MO::MergeDimOp>) {
      axis = splitOrMergeDimOp.getInput().getType().getDims()[i + 1];
    }
    Value splitDim =
        MOGG::lowerDimToKGEN(splitOrMergeDimOp.getLoc(), builder, axis);

    auto callArgs = SmallVector<Value>{kgenIndicesOperand, splitDim};

    return builder
        .create<KGEN::CallOp>(splitOrMergeDimOp.getLoc(),
                              *splitOrMergeDimIndicesFunc, callArgs)
        ->getResult(0);
  }

  /// Given `indices` in KGEN, remap them according to provided `addSingleton`
  /// operation.
  static Value getKGENAddSingletonIndices(
      Value kgenIndicesOperand, MO::AddSingletonDimOp addSingleton,
      MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis, OpBuilder &builder) {
    // construct fully specialized kgen tensor signature
    // (managed_tensor_slice)
    SmallVector<TypedAttr> insertIndexParameterValues = {
        builder.getIndexAttr(
            addSingleton.getInput().getType().getRankNonZero()),
        builder.getIndexAttr(addSingleton.getAxis().getInt()),
        builder.getIndexAttr(0)};

    // Create call to the indices mapping function
    ErrorOr<KGEN::SymbolConstantAttr> insertIndexSymbol =
        mojoLibraryAnalysis.getSpecializedKernel(Utils::InsertIndex,
                                                 insertIndexParameterValues);
    ASSERT_STREAM(succeeded(insertIndexSymbol),
                  << insertIndexSymbol.takeError());

    auto callArgs = SmallVector<Value>{kgenIndicesOperand};

    return builder
        .create<KGEN::CallOp>(addSingleton->getLoc(), *insertIndexSymbol,
                              callArgs)
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
      Value kgenNewOutputTensor, ParamDeclAttr newTensorSpec, ViewOp viewOpUser,
      OpBuilder &builder, MojoLibraryAnalysis &mojoLibraryAnalysis,
      TypeManager &typeManager) {

    MO::TensorType newOutputTensorTy = viewOpUser.getResult().getType();

    mlir::Block &block = outputLambdaToRewrite.getBodyRegion().front();
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

    builder.setInsertionPoint(oldStoreOp);

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
    } else if constexpr (std::is_same_v<ViewOp, MO::SplitDimOp> ||
                         std::is_same_v<ViewOp, MO::MergeDimOp>) {
      kgenNewIndicesOperand = getKGENSplitOrMergeDimIndices(
          kgenIndicesOperand, viewOpUser, mojoLibraryAnalysis, builder);
    }
    ASSERT_STREAM(kgenNewIndicesOperand,
                  << "The new indices for this affine mapping were not "
                     "found! Likely using unsupported view op type: "
                  << viewOpUser);

    ASSERT_STREAM(outputLambdaToRewrite.getInputParams().size() == 2,
                  "Expect output lambda to only declare two parameters "
                  "(simd width) and (element_alignment)");
    ParamDeclAttr simdWidthAttr = outputLambdaToRewrite.getInputParams()[0];
    ParamDeclAttr elementAlignmentAttr =
        outputLambdaToRewrite.getInputParams()[1];
    ASSERT_STREAM(isa<mlir::IndexType>(simdWidthAttr.getType()),
                  << "Expect output lambda to only declare one parameter "
                     "(simd width)");

    // construct fully specialized kgen tensor signature
    // (managed_tensor_slice)

    auto strideVals = MOGG::getStridesForTensor(newOutputTensorTy, false);
    auto stridesOrError = typeManager.lowerStaticDims(builder, strideVals);
    ASSERT_STREAM(!stridesOrError, << "expected stride calculation to work!");
    SmallVector<TypedAttr> simdStoreInternalParameterValues = {
        typeManager.lookupTypeBinding(newOutputTensorTy.getDtype()), // type
        builder.getIndexAttr(newOutputTensorTy.getRank()),           // rank
        KGEN::ParamDeclRefAttr::get(simdWidthAttr), // simd_width
        KGEN::ParamDeclRefAttr::get(newTensorSpec),
        KGEN::ParamDeclRefAttr::get(elementAlignmentAttr), // element_alignment
        builder.getBoolAttr(
            KGEN::MOGGPreElab::kIOSpecImmutable), // mut (of IOSpec)
        builder.getIndexAttr(
            KGEN::MOGGPreElab::kIOSpecIOUnknown), // input (of IOSpec)
    };

    auto simdStoreInternalSymbol = mojoLibraryAnalysis.getSpecializedKernel(
        Utils::DPSManagedTensorSliceStoreSIMD,
        simdStoreInternalParameterValues);
    ASSERT_STREAM(succeeded(simdStoreInternalSymbol),
                  << simdStoreInternalSymbol.takeValue());

    auto signature = simdStoreInternalSymbol->getType().getBody();

    Value kgenNewIndicesOperandRebound = builder.create<KGEN::RebindOp>(
        kgenNewIndicesOperand.getLoc(), signature.getArguments()[1],
        kgenNewIndicesOperand);

    // Applying a bunch of logic on the parameters
    builder.setInsertionPointAfter(oldStoreOp);

    SmallVector<Value> callArgs = {
        kgenNewOutputTensor, kgenNewIndicesOperandRebound, kgenSIMDOperand};
    builder.create<KGEN::CallOp>(outputLambdaToRewrite->getLoc(),
                                 *simdStoreInternalSymbol, callArgs);

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
        M::MO::bindStaticTensorSpecAttr(
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
    rewriteOutputLambdaInplaceKGEN(
        outputLambdaToRewrite, newResultToKGEN.getResult(), newOutputTensorSpec,
        viewOpUser, builder, mojoLibraryAnalysis, typeManager);

    // Change graph level type of the mogg kernel op and replace the
    // old view.
    result.setType(newOutputTensorType);
    viewOpUser.getResult().replaceAllUsesWith(result);
  }

  template <typename ViewOp>
  void fuseViewOpDim(MOGG::KernelOp kernel, ViewOp viewOp, Value result,
                     MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                     TypeManager &typeManager) {
    fuseViewOpDimKGEN(kernel, viewOp, result, mojoLibraryAnalysis, typeManager);

    // construct the new summary string
    SmallVector<Operation *> ops{kernel, viewOp};
    kernel.setSummary(getSummaryString(ops, FusionType::HARD));
  }

  void runOnOperation() override {
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

    // Repeatedly fuse affine views until convergence.
    bool changed = true;
    while (changed) {
      SmallVector<Operation *> opsToErase;

      auto graph = getOperation();
      graph.walk([&](MOGG::KernelOp kernel) {
        // These fusions fuse `kernel` into users of the kernel for certain ops.
        if (isOutputFusibleNonElementwise(kernel)) {
          Value result = kernel.getResult(0);
          Operation *user = *result.getUsers().begin();
          if (auto addDim = dyn_cast_or_null<MO::AddSingletonDimOp>(user)) {
            fuseViewOpDim(kernel, addDim, result, mojoLibraryAnalysis,
                          typeManager);
            opsToErase.push_back(user);
          } else if (auto split = dyn_cast_or_null<MO::SplitDimOp>(user)) {
            fuseViewOpDim(kernel, split, result, mojoLibraryAnalysis,
                          typeManager);
            opsToErase.push_back(user);
          } else if (auto merge = dyn_cast_or_null<MO::MergeDimOp>(user)) {
            fuseViewOpDim(kernel, merge, result, mojoLibraryAnalysis,
                          typeManager);
            opsToErase.push_back(user);
          }
        }

        MOGG::addForwardDeclsOfReferencedFuncs(
            kernel, kernel->getParentOfType<ModuleOp>(), mojoLibraryAnalysis);
      });

      // Check if we did anything
      changed = !opsToErase.empty();

      // Delete all fused view ops
      for (Operation *oldViewOp : opsToErase)
        oldViewOp->erase();

      // Run CSE pass prior to the next iteration.
      if (changed) {
        IRRewriter rewriter(&getContext());
        mlir::DominanceInfo domInfo;
        eliminateCommonSubExpressions(rewriter, domInfo, graph);
      }
    }
  }
};

} // namespace
} // namespace M::MOGG
