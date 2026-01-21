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
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"
#include "KGEN/MOGGPreElab/MOGGPreElabHelpers.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "Support/AssertStream.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/CSE.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEAFFINEVIEWS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// TODO: Better to rename the pass, by no means a gather operation (which is
// handled in the pass) is "affine".
struct FuseAffineViewsPass
    : public impl::FuseAffineViewsBase<FuseAffineViewsPass> {
  using Base::Base;

  static inline bool isFusibleNonElementwise(ExpKernelOp kernel) {
    // Cannot be elementwise kernel, that is fused differently.
    auto isElemwise = kernel.getElemwiseBodyLambda();
    if (isElemwise)
      return false;

    return true;
  }

  static inline bool isOutputFusibleNonElementwise(ExpKernelOp kernel) {
    if (kernel->getNumResults() != 1)
      return false;

    if (!isFusibleNonElementwise(kernel))
      return false;

    if (!kernel.getBoundOutputLambda(0))
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

  static Value lowerScalarDim(Location &loc, OpBuilder builder, TypedAttr dim) {
    if (auto intAttr = dyn_cast<IntegerAttr>(dim)) {
      auto scalarTy = MO::ScalarType::get(intAttr.getContext(),
                                          *MO::fromMLIRType(intAttr.getType()));
      return MOGG::ConstantScalarOp::create(builder, loc, scalarTy, intAttr);
    } else {
      // TODO: We should probably update GetDimOp to directly take the
      // `KGEN::ParamDeclRefAttr` instead of a string.
      return MOGG::GetDimOp::create(
          builder, loc, cast<KGEN::ParamDeclRefAttr>(dim).getName());
    }
  }

  // Support fusion for new-style kernels which go directly to KGEN ops
  template <typename ViewOp>
  void fuseViewOpDimKGEN(MOGG::ExpKernelOp kernel, ViewOp viewOpUser,
                         Value result) {
    mlir::IRRewriter builder(kernel.getContext());

    // Save the old output placeholder and associated type
    OutputPlaceholdersOp oldOutputPlaceholder = kernel.getOutputPlaceholder();

    // Create new output placeholder op which points to the new affine shape
    MO::TensorType newOutputTensorType = viewOpUser.getResult().getType();
    SmallVector<Type> newOutputPlaceHolderTy = {newOutputTensorType};
    builder.setInsertionPoint(oldOutputPlaceholder);
    OutputPlaceholdersOp newOutputPlaceholder = OutputPlaceholdersOp::create(
        builder, oldOutputPlaceholder->getLoc(), newOutputPlaceHolderTy);

    BindLambdaOp outputLambda = kernel.getBoundOutputLambda(0);
    outputLambda->setOperand(0, newOutputPlaceholder.getResult(0));
    Block &outputLambdaBody = outputLambda.getLambda().front();
    auto outputOp = cast<MOGG::OutputOp>(outputLambdaBody.getTerminator());
    auto indexValue = outputOp.getOperand(1);
    auto indexValueType = cast<KGEN::POP::ArrayType>(indexValue.getType());
    auto newIndexValueType = KGEN::POP::ArrayType::get(
        viewOpUser.getType().getRank(), indexValueType.getElementType());

    mlir::NamedAttrList parameters;
    parameters.set("axis", viewOpUser.getAxisAttr());
    SmallVector<Value> operands = {indexValue};
    StringLiteral funcName = "";
    Location loc = viewOpUser.getLoc();
    if constexpr (std::is_same_v<ViewOp, MO::SplitDimOp>) {
      TypedAttr axis =
          viewOpUser.getType()
              .getDims()[viewOpUser.getAxis().getLimitedValue() + 1];
      operands.push_back(lowerScalarDim(loc, builder, axis));
      funcName = Utils::SplitDimIndices;
    } else if (std::is_same_v<ViewOp, MO::MergeDimOp>) {
      TypedAttr axis =
          viewOpUser.getInput()
              .getType()
              .getDims()[viewOpUser.getAxis().getLimitedValue() + 1];
      operands.push_back(lowerScalarDim(loc, builder, axis));
      funcName = Utils::MergeDimIndices;
    } else if (std::is_same_v<ViewOp, MO::AddSingletonDimOp>) {
      parameters.set("value", builder.getIndexAttr(0));
      funcName = Utils::InsertIndex;
    }
    ASSERT_STREAM(!funcName.empty(), << "Unsupported affine view op");

    builder.setInsertionPoint(outputOp);
    auto newIndexValue =
        MOGG::CallOp::create(builder, loc, TypeRange{newIndexValueType},
                             builder.getStringAttr(funcName), operands,
                             mlir::ArrayAttr{},
                             builder.getDictionaryAttr(parameters))
            .getResult(0);
    outputOp->setOperand(1, newIndexValue);

    // Change graph level type of the mogg kernel op and replace the
    // old view.
    result.setType(newOutputTensorType);
    viewOpUser.getResult().replaceAllUsesWith(result);
  }

  template <typename ViewOp>
  void fuseViewOpDim(MOGG::ExpKernelOp kernel, ViewOp viewOp, Value result) {
    fuseViewOpDimKGEN(kernel, viewOp, result);

    // construct the new summary string
    SmallVector<Operation *> ops{kernel, viewOp};
    kernel.setSummary(getSummaryString(ops, FusionType::HARD));
  }

  void runOnOperation() override {
    // Repeatedly fuse affine views until convergence.
    bool changed = true;
    while (changed) {
      SmallVector<Operation *> opsToErase;

      auto graph = getOperation();
      graph.walk([&](MOGG::ExpKernelOp kernel) {
        // These fusions fuse `kernel` into users of the kernel for certain ops.
        if (isOutputFusibleNonElementwise(kernel)) {
          Value result = kernel.getResult(0);
          Operation *user = *result.getUsers().begin();
          if (auto addDim = dyn_cast_or_null<MO::AddSingletonDimOp>(user)) {
            fuseViewOpDim(kernel, addDim, result);
            opsToErase.push_back(user);
          } else if (auto split = dyn_cast_or_null<MO::SplitDimOp>(user)) {
            fuseViewOpDim(kernel, split, result);
            opsToErase.push_back(user);
          } else if (auto merge = dyn_cast_or_null<MO::MergeDimOp>(user)) {
            fuseViewOpDim(kernel, merge, result);
            opsToErase.push_back(user);
          }
        }
      });

      // Check if we did anything
      changed = !opsToErase.empty();
      numberOfFusedViews += opsToErase.size();

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
