//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "Support/AssertStream.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/CSE.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSERESHAPESINTOEPILOGUES
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

// Helper functions for reshape-into-epilogue fusion (internal to this
// translation unit). Defined at file scope so they can be called by both the
// pass struct and the exported fuseReshapesIntoEpiloguesStep function.
namespace {

// Note: this pass handles reshape ops (SplitDim, MergeDim, AddSingletonDim)
// by fusing them into kernel output lambdas as affine index transforms.

static bool isFusibleNonElementwise(M::MOGG::KernelOp kernel) {
  // Cannot be elementwise kernel, that is fused differently.
  auto isElemwise = kernel.getElemwiseBodyLambda();
  if (isElemwise)
    return false;

  return true;
}

static bool isOutputFusibleNonElementwise(M::MOGG::KernelOp kernel) {
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
  if (auto intAttr = dyn_cast<KGEN::SIMDAttr>(dim)) {
    auto scalarTy = M::MO::ScalarType::get(
        intAttr.getContext(), *intAttr.getType().getResolvedDType());
    return M::MOGG::ConstantScalarOp::create(builder, loc, scalarTy, intAttr);
  } else {
    // TODO: We should probably update GetDimOp to directly take the
    // `KGEN::ParamDeclRefAttr` instead of a string.
    return M::MOGG::GetDimOp::create(
        builder, loc, cast<KGEN::ParamDeclRefAttr>(dim).getName());
  }
}

// Support fusion for new-style kernels which go directly to KGEN ops
template <typename ViewOp>
static void fuseViewOpDimKGEN(M::MOGG::KernelOp kernel, ViewOp viewOpUser,
                              Value result,
                              const M::MOGG::MojoRegistry &registry) {
  mlir::IRRewriter builder(kernel.getContext());

  // Save the old output placeholder and associated type
  M::MOGG::OutputPlaceholdersOp oldOutputPlaceholder =
      kernel.getOutputPlaceholder();

  // Create new output placeholder op which points to the new affine shape
  M::MO::TensorType newOutputTensorType = viewOpUser.getResult().getType();
  SmallVector<Type> newOutputPlaceHolderTy = {newOutputTensorType};
  builder.setInsertionPoint(oldOutputPlaceholder);
  M::MOGG::OutputPlaceholdersOp newOutputPlaceholder =
      M::MOGG::OutputPlaceholdersOp::create(
          builder, oldOutputPlaceholder->getLoc(), newOutputPlaceHolderTy);

  M::MOGG::BindLambdaOp outputLambda = kernel.getBoundOutputLambda(0);
  outputLambda->setOperand(0, newOutputPlaceholder.getResult(0));
  Block &outputLambdaBody = outputLambda.getLambda().front();
  auto outputOp = cast<M::MOGG::OutputOp>(outputLambdaBody.getTerminator());
  auto indexValue = outputOp.getOperand(1);
  auto indexValueType = cast<KGEN::POP::ArrayType>(indexValue.getType());
  auto newIndexValueType = KGEN::POP::ArrayType::get(
      viewOpUser.getType().getRank(), indexValueType.getElementType());

  mlir::NamedAttrList parameters;
  parameters.set("axis", viewOpUser.getAxisAttr());
  SmallVector<Value> operands = {indexValue};
  Location loc = viewOpUser.getLoc();
  if constexpr (std::is_same_v<ViewOp, M::MO::SplitDimOp>) {
    TypedAttr axis = viewOpUser.getType()
                         .getDims()[viewOpUser.getAxis().getLimitedValue() + 1];
    operands.push_back(lowerScalarDim(loc, builder, axis));
  } else if constexpr (std::is_same_v<ViewOp, M::MO::MergeDimOp>) {
    TypedAttr axis = viewOpUser.getInput()
                         .getType()
                         .getDims()[viewOpUser.getAxis().getLimitedValue() + 1];
    operands.push_back(lowerScalarDim(loc, builder, axis));
  } else if constexpr (std::is_same_v<ViewOp, M::MO::AddSingletonDimOp>) {
    parameters.set("value", builder.getIndexAttr(0));
  }
  StringRef funcName = viewOpUser.getOperationName();
  M::MOGG::PrimitiveDeclOp primitive = registry.lookupPrimitive(funcName);

  builder.setInsertionPoint(outputOp);
  auto newIndexValue =
      M::MOGG::CallAffineOp::create(builder, loc, newIndexValueType,
                                    primitive.getSymNameAttr(), operands,
                                    builder.getDictionaryAttr(parameters))
          .getResult();
  outputOp->setOperand(1, newIndexValue);

  // Change graph level type of the mogg kernel op and replace the
  // old view.
  result.setType(newOutputTensorType);
  viewOpUser.getResult().replaceAllUsesWith(result);

  // The old placeholder is no longer referenced by the output lambda.
  // Erase it so the kernel has exactly one output placeholder, as required
  // by getOutputPlaceholder().
  oldOutputPlaceholder->erase();
}

template <typename ViewOp>
static void fuseViewOpDim(M::MOGG::KernelOp kernel, ViewOp viewOp, Value result,
                          const M::MOGG::MojoRegistry &registry) {
  fuseViewOpDimKGEN(kernel, viewOp, result, registry);

  SmallVector<Operation *> ops{kernel, viewOp};
  kernel.setFusionInfoAttr(M::MOGG::createFusionInfo<M::MOGG::FusionAffineAttr>(
      kernel->getContext(), ops));
  // Merge source locations so that Python stack traces from fused-in ops
  // survive into AddStackTracesToAssertsPass.
  kernel->setLoc(mlir::FusedLoc::get(kernel->getContext(),
                                     {viewOp->getLoc(), kernel->getLoc()}));
}

} // namespace

// Exported shared function: one iteration of reshape-into-epilogue fusion.
// Returns the number of views fused. Called by both FuseReshapesIntoEpilogues
// and FuseEpilogues.
size_t M::MOGG::fuseReshapesIntoEpiloguesStep(MO::GraphOp graph,
                                              const MojoRegistry &registry) {
  SmallVector<Operation *> opsToErase;

  graph.walk([&](MOGG::KernelOp kernel) {
    if (isOutputFusibleNonElementwise(kernel)) {
      Value result = kernel.getResult(0);
      Operation *user = *result.getUsers().begin();
      if (auto addDim = dyn_cast_or_null<MO::AddSingletonDimOp>(user)) {
        fuseViewOpDim(kernel, addDim, result, registry);
        opsToErase.push_back(user);
      } else if (auto split = dyn_cast_or_null<MO::SplitDimOp>(user)) {
        fuseViewOpDim(kernel, split, result, registry);
        opsToErase.push_back(user);
      } else if (auto merge = dyn_cast_or_null<MO::MergeDimOp>(user)) {
        fuseViewOpDim(kernel, merge, result, registry);
        opsToErase.push_back(user);
      }
    }
  });

  for (Operation *oldViewOp : opsToErase)
    oldViewOp->erase();

  return opsToErase.size();
}

namespace M::MOGG {
namespace {

struct FuseReshapesIntoEpiloguesPass
    : public impl::FuseReshapesIntoEpiloguesBase<
          FuseReshapesIntoEpiloguesPass> {
  using Base::Base;

  void runOnOperation() override {
    auto graph = getOperation();
    MojoRegistry registry(graph->getParentOfType<mlir::ModuleOp>());
    size_t totalFused = 0;
    size_t fused;
    do {
      fused = fuseReshapesIntoEpiloguesStep(graph, registry);
      totalFused += fused;
      if (fused > 0) {
        IRRewriter rewriter(&getContext());
        mlir::DominanceInfo domInfo;
        eliminateCommonSubExpressions(rewriter, domInfo, graph);
      }
    } while (fused > 0);
    numberOfFusedViews += totalFused;
  }
};

} // namespace
} // namespace M::MOGG
