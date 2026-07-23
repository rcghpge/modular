//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/AssertStream.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Pass/Pass.h"

namespace M::MO {

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_DECOMPOSEVIEWOPS
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

using namespace M;
using namespace M::MO;

namespace {

// Decomposition for flattening ND -> 1D
// e.g: [a, b, c] -> [a*b*c] via N-1 merge_dim ops
// Requires that all dims are static
static void decomposeFlatteningReshape(Static::ReshapeOp reshape,
                                       OpBuilder &builder) {
  auto oldOutType = reshape.getResult().getType();
  [[maybe_unused]] auto inRank = reshape.getInput().getType().getRank();
  [[maybe_unused]] auto outRank = reshape.getType().getRank();
  ASSERT_STREAM(inRank > 1 && outRank == 1,
                "Expected a reshape to be a flattening reshape");

  MergeDimOp lastMergeOp;
  auto rollingShape = reshape.getInput().getType().cloneWithoutLayout();

  for (size_t i = 0; i < inRank - 1; ++i) {
    auto shapeAttr = rollingShape.getTypedShapeAttr();
    SmallVector<TypedAttr> newShape{shapeAttr.getValues()};

    auto dimIdx = shapeAttr.getSize() - 1;
    auto lastDim = shapeAttr[dimIdx];
    auto secondLastDim = shapeAttr[dimIdx - 1];

    // remove the last two dims from the shape ([..., a, b, c] -> [a])
    newShape.pop_back();
    newShape.pop_back();

    // add back the multiplied dim ([..., a] -> [a, b*c])
    SmallVector<TypedAttr> dimsToMultiply{lastDim, secondLastDim};
    TypedAttr multipliedDim =
        KGEN::ParamOperatorAttr::get(KGEN::POC::MulNoWrap, dimsToMultiply);
    newShape.push_back(multipliedDim);

    // create the new merge_dim op using the new shape
    rollingShape = rollingShape.cloneWith(newShape);
    auto mergeOp = MergeDimOp::create(
        builder, reshape.getLoc(), rollingShape, reshape.getInput(),
        /*axis*/ builder.getIndexAttr(dimIdx - 1));

    reshape.setOperand(mergeOp.getResult());
    lastMergeOp = mergeOp;
  }

  // Make sure the layout info for the last merge_dim op matches the output
  // of the reshape
  auto mergeOpResult = lastMergeOp.getResult();
  auto tt = mergeOpResult.getType();
  if (auto outLayout = reshape.getType().getLayoutAttr())
    mergeOpResult.setType(tt.cloneWith(outLayout));
  else
    mergeOpResult.setType(tt.cloneWithoutLayout());

  // If the final output type doesn't match the input type, we need to rebind
  // the final output to the input type.
  // e.g: [mul_no_wrap(D1, D2, D3)] -> [D123]
  auto finalOutput = lastMergeOp.getResult();
  if (finalOutput.getType() != oldOutType) {
    auto rebindOp =
        RebindOp::create(builder, reshape.getLoc(), oldOutType, finalOutput);
    finalOutput = rebindOp.getResult();
  }

  // delete the reshape
  reshape.replaceAllUsesWith(finalOutput);
  reshape.erase();
}

// Decomposition for unflattening 1D -> ND
// e.g: [a*b*c] -> [a, b, c] via N-1 split_dim ops
// Requires that all dims are static
static void decomposeUnflatteningReshape(Static::ReshapeOp reshape,
                                         OpBuilder &builder) {
  auto oldOutType = reshape.getResult().getType();
  [[maybe_unused]] auto inRank = reshape.getInput().getType().getRank();
  auto outRank = reshape.getType().getRank();
  ASSERT_STREAM(inRank == 1 && outRank > 1,
                "Expected a reshape to be an unflattening reshape");

  auto outputShapeAttr = reshape.getType().getTypedShapeAttr();
  SplitDimOp lastSplitOp;
  auto rollingShape = reshape.getInput().getType().cloneWithoutLayout();

  for (size_t i = 0; i < outRank - 1; ++i) {
    auto shapeAttr = rollingShape.getTypedShapeAttr();
    SmallVector<TypedAttr> newShape{shapeAttr.getValues()};

    auto dimIdx = shapeAttr.getSize() - 1;
    auto lastDim = shapeAttr[dimIdx];
    auto currDimOfOutput = outputShapeAttr[dimIdx];

    // remove the last dim from the shape ([..., a*b*c] -> [...])
    newShape.pop_back();

    // add back two dims ([...] -> [..., a, b*c])
    SmallVector<TypedAttr> dimsToDivide{lastDim, currDimOfOutput};
    TypedAttr remaining =
        KGEN::ParamOperatorAttr::get(KGEN::POC::Div, dimsToDivide);
    // If this results in a divide that is not simplified out, we need to handle
    // the case where `currDimOfOutput` is zero.
    if (!MOSH::ShapeAttr::getStaticDim(currDimOfOutput).has_value()) {
      if (auto algebraicDim = dyn_cast<KGEN::ParamOperatorAttr>(remaining)) {
        if (algebraicDim.getOpcode() == KGEN::POC::Div) {
          auto operands = algebraicDim.getOperands();
          ASSERT_STREAM(operands.size() == 2,
                        "Expected a divide to have two operands");
          auto divisor = operands[1];
          if (divisor == currDimOfOutput) {
            // Add a guard to the divide to handle the case where the divisor is
            // zero.
            //
            // Note that KGEN::POC::Cond lowers to pop.select which evaluates
            // both sides of the condition.
            //
            // This means that this naive expression may still yield a division
            // by zero:
            //   `cond(eq(D2, 0), 0, div(D1, D2))`
            // To remedy this, we generate a much uglier expression:
            //   `cond(eq(D2, 0), 0, div(D1, cond(eq(D2, 0), 1, D2)))`
            TypedAttr zero = MOSH::ShapeAttr::getDim(builder.getContext(), 0);
            TypedAttr one = MOSH::ShapeAttr::getDim(builder.getContext(), 1);

            auto isDivisorZero =
                KGEN::ParamOperatorAttr::get(KGEN::POC::EQ, {divisor, zero});
            auto nonZeroDivisor = KGEN::ParamOperatorAttr::get(
                KGEN::POC::Cond, {isDivisorZero, one, divisor});
            auto safeDivision = KGEN::ParamOperatorAttr::get(
                KGEN::POC::Div, {lastDim, nonZeroDivisor});
            remaining = KGEN::ParamOperatorAttr::get(
                KGEN::POC::Cond, {isDivisorZero, zero, safeDivision});
          }
        }
      }
    }

    newShape.push_back(currDimOfOutput);
    newShape.push_back(remaining);

    // create the new split_dim op using the new shape
    rollingShape = rollingShape.cloneWith(newShape);
    auto splitOp = SplitDimOp::create(builder, reshape.getLoc(), rollingShape,
                                      reshape.getInput(),
                                      /*axis*/ builder.getIndexAttr(dimIdx));

    reshape.setOperand(splitOp.getResult());
    lastSplitOp = splitOp;
  }

  // Make sure the layout info for the last split_dim op matches the output
  // of the reshape
  auto splitOpResult = lastSplitOp.getResult();
  auto tt = splitOpResult.getType();
  if (auto outLayout = reshape.getType().getLayoutAttr())
    splitOpResult.setType(tt.cloneWith(outLayout));
  else
    splitOpResult.setType(tt.cloneWithoutLayout());

  // If the final output type doesn't match the input type, we need to rebind
  // the final output to the input type.
  // e.g: [mul_no_wrap(D1, D2, D3)] -> [D123]
  auto finalOutput = lastSplitOp.getResult();
  if (finalOutput.getType() != oldOutType) {
    auto rebindOp =
        RebindOp::create(builder, reshape.getLoc(), oldOutType, finalOutput);
    finalOutput = rebindOp.getResult();
  }

  // delete the reshape
  reshape.replaceAllUsesWith(finalOutput);
  reshape.erase();
}

static bool isStaticShapeWithZeroElts(MOSH::ShapeAttr shapeAttr) {
  if (!shapeAttr.isStatic())
    return false;

  auto numElts = shapeAttr.getSymbolicNumElements();
  auto asIntVal = MOSH::ShapeAttr::getStaticDim(numElts);
  return asIntVal == 0;
}

static void decomposeReshape(Static::ReshapeOp reshape, OpBuilder &builder) {
  builder.setInsertionPointAfter(reshape);
  TensorType outType = reshape.getType();

  MOSH::ShapeAttr inShapeAttr =
      reshape.getInput().getType().getTypedShapeAttr();
  MOSH::ShapeAttr outShapeAttr = outType.getTypedShapeAttr();
  size_t inRank = reshape.getInput().getType().getRank();
  size_t outRank = reshape.getType().getRank();

  // The following decompositions require that all dims are known at compile
  // time and that the number of elements is non-zero.
  if (!isStaticShapeWithZeroElts(inShapeAttr) &&
      !isStaticShapeWithZeroElts(outShapeAttr)) {
    // Ultra specific decompositions for flattening and unflattening
    if (inRank > 1 && outRank == 1) {
      decomposeFlatteningReshape(reshape, builder);
      return;
    } else if (inRank == 1 && outRank > 1) {
      decomposeUnflatteningReshape(reshape, builder);
      return;
    }
  }

  // Remove all `1`s from the reshape by adding a singleton dimension after.
  SmallVector<TypedAttr> newOutShape;
  for (size_t idx = 0; idx < outShapeAttr.getSize(); ++idx) {
    TypedAttr dimVal = outShapeAttr[idx];
    auto asInt = MOSH::ShapeAttr::getStaticDim(dimVal);
    if (!asInt || *asInt != 1)
      newOutShape.push_back(dimVal);
  }

  TensorType outShapeWithoutOnes = outType.cloneWith(newOutShape);

  TypedValue<TensorType> output = reshape.getResult();
  output.setType(outShapeWithoutOnes);

  // The shape to progressively add the ones back in.
  TensorType rollingShape = outShapeWithoutOnes;

  // Create a new reshape built up from just these new ranks.
  for (size_t idx = 0; idx < outShapeAttr.getSize(); ++idx) {
    TypedAttr dimVal = outShapeAttr[idx];
    auto asInt = MOSH::ShapeAttr::getStaticDim(dimVal);
    if (!asInt || *asInt != 1)
      continue;

    MOSH::ShapeAttr shapeAttr = rollingShape.getTypedShapeAttr();
    SmallVector<TypedAttr> newShape{shapeAttr.getValues()};
    newShape.insert(newShape.begin() + idx,
                    MOSH::ShapeAttr::getDim(builder.getContext(), 1));
    rollingShape = rollingShape.cloneWith(newShape);

    auto op = AddSingletonDimOp::create(builder, reshape.getLoc(), rollingShape,
                                        output,
                                        /*axis=*/builder.getIndexAttr(idx));

    output.replaceAllUsesExcept(op.getResult(), op);
    output = op.getResult();
  }

  // If the reshape is now a no-op, remove it.
  if (reshape.getType() == reshape.getInput().getType()) {
    reshape.getResult().replaceAllUsesWith(reshape.getInput());
    reshape.erase();
    return;
  }

  size_t newInRank = reshape.getInput().getType().getRank();
  size_t newOutRank = reshape.getType().getRank();

  // Try split dimension. Super specific for 2D -> 3D split for now.
  if (newInRank == 2 && newOutRank == 3) {
    MOSH::ShapeAttr outShapeAttr = reshape.getType().getTypedShapeAttr();
    MOSH::ShapeAttr inShapeAttr =
        reshape.getInput().getType().getTypedShapeAttr();

    // Pattern match [N, K] -> [n1, n2, K]
    if (outShapeAttr[0] != inShapeAttr[0] &&
        outShapeAttr[1] != inShapeAttr[1] &&
        outShapeAttr[1] != inShapeAttr[0] &&
        outShapeAttr[2] == inShapeAttr[1]) {

      Value split = SplitDimOp::create(builder, reshape.getLoc(),
                                       reshape.getType(), reshape.getInput(),
                                       /*axis*/ builder.getIndexAttr(0));
      reshape.replaceAllUsesWith(split);
      reshape.erase();
    }
  }
}

struct DecomposeViewsPass
    : public impl::DecomposeViewOpsBase<DecomposeViewsPass> {
  using DecomposeViewOpsBase::DecomposeViewOpsBase;

  void runOnOperation() override {
    OpBuilder builder{&getContext()};
    getOperation().walk([&](Operation *op) {
      builder.setInsertionPoint(op);

      if (auto reshape = dyn_cast<Static::ReshapeOp>(op))
        decomposeReshape(reshape, builder);
    });
  }
};

} // namespace
