//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Implementations of `Staticization` interface for various `mo` ops.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/ShapeFunctions.h"
#include "GenericML/Support/Dims.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include <sstream>

namespace M::MO {

namespace {

/// Staticization implementation for `mo.reshape` and `mo.static.reshape`.
static TypedAttr tryStaticizeReshapeLikeOp(TensorType inputTy,
                                           TensorType outputTy,
                                           ArrayRef<TypedAttr> inputs,
                                           ParamExprBuilder &b) {
  // Only handle ranked input and output shapes.
  auto inputShape = dyn_cast<ShapeAttr>(inputTy.getShapeAttr());
  auto outputShape = dyn_cast<ShapeAttr>(outputTy.getShapeAttr());
  if (!inputShape || !outputShape)
    return TypedAttr{};
  size_t inputRank = inputShape.getRank();
  size_t outputRank = outputShape.getRank();

  // Check whether the given attribute is a constant integer 1.
  auto isConstantOne = [&](TypedAttr attr) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getInt() == 1;
    return false;
  };

  // Grab staticized version of input.
  TypedAttr inputAttr = inputs[0];

  // Handle [1] -> [] case.
  if (inputRank == 1 && isConstantOne(inputShape[0]) && outputRank == 0)
    return inputAttr;

  // Handle [] -> [1] case.
  if (inputRank == 0 && outputRank == 1 && isConstantOne(outputShape[0]))
    return b.getShapeAttr({inputAttr});

  return TypedAttr{};
}

} // namespace

ErrorOr<TypedAttr> BroadcastShapeOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                                  ParamExprBuilder &b) {

  // Requires inputs to be lists of param exprs.
  auto lhsShapeAttr = dyn_cast<ShapeAttr>(inputs[0]);
  auto rhsShapeAttr = dyn_cast<ShapeAttr>(inputs[1]);
  if (lhsShapeAttr && rhsShapeAttr) {

    // Ensure rank(lhsShapeAttr) >= rank(rhsShapeAttr)
    if (lhsShapeAttr.getRank() < rhsShapeAttr.getRank())
      std::swap(lhsShapeAttr, rhsShapeAttr);

    // Build up output value.
    SmallVector<TypedAttr> resDims;
    size_t rankDiff = lhsShapeAttr.getRank() - rhsShapeAttr.getRank();
    for (size_t axis = 0; axis < lhsShapeAttr.getRank(); axis++) {
      if (axis < rankDiff) {
        // Extra leading dimensions get broadcasted trivially.
        resDims.push_back(lhsShapeAttr[axis]);

      } else {
        // Otherwise this dimension is broadcasted by the following rule:
        //   def bcast(D1, D2):
        //       assert `D1 == D2 or D1 == 1 or D2 == 1`
        //       if D1 == 1:
        //           return D2
        //       else:
        //           return D1
        Var dim1 = b.createVar(lhsShapeAttr[axis]);
        Var dim2 = b.createVar(rhsShapeAttr[axis - rankDiff]);
        // Insert assertion
        b.createAssert(dim1 == 1 || dim2 == 1 || dim1 == dim2,
                       "[broadcast_shape] dimensions are not broadcastable");

        Var newResDim = b.createCond(dim1 == 1, dim2, dim1);
        resDims.push_back(newResDim.getAttr());
      }
    }
    auto res = MOSH::ShapeAttr::get(resDims, lhsShapeAttr.getType());
    return res;
  }

  // Can't staticize.
  return TypedAttr{};
}

ErrorOr<TypedAttr> CastOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                        ParamExprBuilder &b) {
  // Only allow int to int cast for now.
  DType sourceDtype = getInput().getType().getDtype();
  DType targetDtype = getType().getDtype();
  if (sourceDtype.isInt() && targetDtype.isInt())
    return inputs[0];

  return TypedAttr{};
}

ErrorOr<TypedAttr> ConcatOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                          ParamExprBuilder &b) {
  // We only staticize over 1D tensors for now, so axis must be 0.
  if (tryGetConstantInt(inputs[0]) != 0)
    return TypedAttr{};

  // Build up staticized version of values from individual inputs (ignoring the
  // first axis input).
  SmallVector<TypedAttr> outVals;
  for (TypedAttr input : inputs.drop_front()) {
    if (auto inputShapeAttr = dyn_cast<ShapeAttr>(input)) {
      // Accumulate staticized values from this input.
      llvm::append_range(outVals, inputShapeAttr.getValues());

    } else {
      // Give up if any input can't be staticized as a 1D tensor.
      return TypedAttr{};
    }
  }

  return b.getShapeAttr(outVals);
}

ErrorOr<TypedAttr> GatherOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                          ParamExprBuilder &b) {
  // We only staticize over 1D tensors for now, so axis must be 0.
  if (tryGetConstantInt(inputs[2]) != 0)
    return TypedAttr{};

  // We require the input to be staticized to a list of symbolic values.
  auto inputElems = dyn_cast<ShapeAttr>(inputs[0]);
  if (!inputElems)
    return TypedAttr{};
  int64_t numInputElems = inputElems.getSize();

  // Collect constant indices.
  SmallVector<int64_t> indices;
  bool isIndicesScalar = false;
  if (auto indicesOr = tryGetConstantIntVector(inputs[1])) {
    indices = indicesOr.value();

  } else if (auto idxOr = tryGetConstantInt(inputs[1])) {
    indices.push_back(idxOr.value());
    isIndicesScalar = true;

  } else {
    // Give up if indices isn't a constant scalar/1D tensor.
    return TypedAttr{};
  }

  // Normalize & validate individual index, and collect the corresponding
  // input elements.
  SmallVector<TypedAttr> outElems;
  for (int64_t idx : indices) {
    int64_t normalizedIdx = normalizeAxis(idx, numInputElems);
    if (normalizedIdx < 0 || numInputElems <= normalizedIdx) {
      std::stringstream ss;
      ss << "[gather] index out of bound: " << idx
         << " for input size: " << numInputElems;
      return Error(ss.str());
    }
    outElems.push_back(inputElems[normalizedIdx]);
  }

  // Return a dimension attribute or shape attribute based on whether indices
  // was a scalar or 1D tensor.
  if (isIndicesScalar)
    return outElems.front();
  return b.getShapeAttr(outElems);
}

ErrorOr<TypedAttr> RebindOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                          ParamExprBuilder &b) {
  return inputs[0];
}

ErrorOr<TypedAttr> ReduceMulOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                             ParamExprBuilder &b) {
  // We only staticize over 1D tensors for now, so axis must be 0.
  if (tryGetConstantInt(inputs[1]) != 0)
    return TypedAttr{};

  // The result is just the product of individual symbolic elements from the
  // staticized input.
  auto inputElems = dyn_cast<ShapeAttr>(inputs[0]);
  if (!inputElems)
    return TypedAttr{};

  TypedAttr productAttr =
      b.createDeclToKgenExpr(KGEN::POC::MulNuw, inputElems.getValues());
  return b.getShapeAttr({productAttr});
}

ErrorOr<TypedAttr> ReshapeOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                           ParamExprBuilder &b) {
  return tryStaticizeReshapeLikeOp(getInput().getType(), getType(), inputs, b);
}

ErrorOr<TypedAttr> Static::ReshapeOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                                   ParamExprBuilder &b) {
  return tryStaticizeReshapeLikeOp(getInput().getType(), getType(), inputs, b);
}

} // namespace M::MO
