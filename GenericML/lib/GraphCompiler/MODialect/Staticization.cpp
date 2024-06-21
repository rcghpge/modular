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
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
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
  if (inputRank == 1 && isConstantOne(inputShape[0]) && outputRank == 0) {
    if (auto inputElems = dyn_cast<ShapeAttr>(inputAttr))
      return inputElems[0];

  } else if (inputRank == 0 && outputRank == 1 &&
             isConstantOne(outputShape[0])) {
    // Handle [] -> [1] case.
    return b.getShapeAttr({inputAttr});
  }

  return TypedAttr{};
}

/// Staticization implementation for elementwise ops (e.g., `mo.add`).  `func`
/// is a function that takes in scalar inputs and return the corresponding
/// version of the output. For instance, assume the underlying op is `mo.add`,
/// given
///   inputs[0] := ShapeAttr([N0, N1])
///   inputs[1] := ShapeAttr([1, 2])
///
/// The result would be
///   ShapeAttr([func({N0, 1}), func({N1, 2}))
static ErrorOr<TypedAttr> tryStaticizeElemWiseOp(
    TensorType outTy, ArrayRef<TypedAttr> inputs, ParamExprBuilder &b,
    llvm::function_ref<TypedAttr(ArrayRef<TypedAttr> scalarInputs)> func) {
  // Give up on dynamically ranked elementwise op.
  if (!outTy.hasRank())
    return TypedAttr{};

  // Handle scalar case.
  if (outTy.getRank() == 0)
    return func(inputs);

  // Handle non-scalar case -- when inputs are lists of param exprs.
  SmallVector<ShapeAttr> inputShapeAttrs;
  for (auto [i, input] : llvm::enumerate(inputs)) {
    // Give up if any input is a shape of a dynamically ranked tensor.
    auto inputShapeAttr = dyn_cast<ShapeAttr>(input);
    if (!inputShapeAttr)
      return TypedAttr{};

    // Make sure all inputs have the same # of elements; this is a sanity check.
    if (!inputShapeAttrs.empty() &&
        inputShapeAttrs.front().getSize() != inputShapeAttr.getSize()) {
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "Cannot staticize binary ops whose input have different number of "
            "elements -- first input: "
         << inputShapeAttrs.front() << ", " << i
         << "-th input: " << inputShapeAttr;
      return Error(msg);
    }
    inputShapeAttrs.push_back(inputShapeAttr);
  }

  // Build up output shape
  SmallVector<TypedAttr> outVals;
  size_t inputNumElems = inputShapeAttrs.front().getSize();
  for (size_t i = 0; i < inputNumElems; i++) {
    SmallVector<TypedAttr> ithIndexScalarInputs;
    for (ShapeAttr inputShapeAttr : inputShapeAttrs)
      ithIndexScalarInputs.push_back(inputShapeAttr[i]);
    outVals.push_back(func(ithIndexScalarInputs));
  }
  return b.getShapeAttr(outVals);
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

ErrorOr<TypedAttr> AddOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                       ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp(getType(), inputs, b, [&](auto scalarInputs) {
    return b.createDeclToKgenExpr(KGEN::POC::Add, scalarInputs);
  });
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

ErrorOr<TypedAttr> DivOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                       ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp(getType(), inputs, b, [&](auto scalarInputs) {
    return b.createDeclToKgenExpr(KGEN::POC::Div, scalarInputs);
  });
}

ErrorOr<TypedAttr> EqualOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                         ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp(getType(), inputs, b, [&](auto scalarInputs) {
    return b.createDeclToKgenExpr(KGEN::POC::EQ, scalarInputs);
  });
}

ErrorOr<TypedAttr> NegativeOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                            ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp(getType(), inputs, b, [&](auto scalarInputs) {
    Var input = b.createVar(scalarInputs[0]);
    Var res = input * -1;
    return res.getAttr();
  });
}

ErrorOr<TypedAttr> MinOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                       ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp(getType(), inputs, b, [&](auto scalarInputs) {
    return b.createDeclToKgenExpr(KGEN::POC::Min, scalarInputs);
  });
}

ErrorOr<TypedAttr> MulOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                       ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp(getType(), inputs, b, [&](auto scalarInputs) {
    return b.createDeclToKgenExpr(KGEN::POC::MulNuw, scalarInputs);
  });
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

ErrorOr<TypedAttr> SelectOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                          ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp(getType(), inputs, b, [&](auto scalarInputs) {
    Var condVar = b.createVar(scalarInputs[0]);
    Var thenVar = b.createVar(scalarInputs[1]);
    Var elseVar = b.createVar(scalarInputs[2]);
    Var res = b.createCond(condVar, thenVar, elseVar);
    return res.getAttr();
  });
}

ErrorOr<TypedAttr> SqueezeShapeOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                                ParamExprBuilder &b) {
  // Make sure inputs aren't too dynamic.
  auto inputShapeAttr = dyn_cast<ShapeAttr>(inputs[0]);
  auto squeezeAxesOr = tryGetConstantIntVector(inputs[1]);
  if (!inputShapeAttr || !squeezeAxesOr.has_value())
    return TypedAttr{};

  // Check, normalize and collect squeeze axes into a set.
  SmallVector<int64_t> &squeezeAxes = squeezeAxesOr.value();
  DenseSet<int64_t> normalizedSqueezeAxisSet;
  int64_t inputShapeLen = inputShapeAttr.getValues().size();
  for (auto &squeezeAxis : squeezeAxes) {
    if (squeezeAxis < -inputShapeLen || squeezeAxis >= inputShapeLen)
      return Error("Squeeze axis not in [-in_rank, in_rank).");

    // normalize negative indexing
    if (squeezeAxis < 0)
      squeezeAxis += inputShapeLen;
    normalizedSqueezeAxisSet.insert(squeezeAxis);
  }

  // Compute the unsqueezed symbolic shape.
  SmallVector<TypedAttr> finalShape;
  for (auto [axis, dimAttr] : llvm::enumerate(inputShapeAttr.getValues())) {
    // Skip if this axis is to be squeezed.
    if (normalizedSqueezeAxisSet.contains(axis)) {
      b.createAssert(b.createVar(dimAttr) == 1,
                     "Attempting to unsqueeze a dimension that is not 1.");
      continue;
    }
    finalShape.push_back(dimAttr);
  }
  auto squeezedShapeAttr = b.getShapeAttr(finalShape);

  return squeezedShapeAttr;
}

ErrorOr<TypedAttr> Static::ReshapeOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                                   ParamExprBuilder &b) {
  return tryStaticizeReshapeLikeOp(getInput().getType(), getType(), inputs, b);
}

ErrorOr<TypedAttr> SubOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                       ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp(getType(), inputs, b, [&](auto scalarInputs) {
    Var lhs = b.createVar(scalarInputs[0]);
    Var rhs = b.createVar(scalarInputs[1]);
    Var res = lhs - rhs;
    return res.getAttr();
  });
}

ErrorOr<TypedAttr> UnsqueezeShapeOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                                  ParamExprBuilder &b) {
  // Make sure inputs aren't too dynamic.
  auto inputShapeAttr = dyn_cast<ShapeAttr>(inputs[0]);
  auto unsqueezeAxesOr = tryGetConstantIntVector(inputs[1]);
  if (!inputShapeAttr || !unsqueezeAxesOr.has_value())
    return TypedAttr{};

  // Check, normalize and collect unsqueeze axes into a set.
  const SmallVector<int64_t> &unsqueezeAxes = unsqueezeAxesOr.value();
  DenseSet<int64_t> unsqueezeAxesSet;
  int64_t outputShapeLen =
      inputShapeAttr.getValues().size() + unsqueezeAxes.size();
  for (auto unsqueezeAxis : unsqueezeAxes) {
    if (unsqueezeAxis < -outputShapeLen || unsqueezeAxis >= outputShapeLen)
      return Error("Unsqueeze axis not in [-out_rank, out_rank).");

    // normalize negative indexing
    if (unsqueezeAxis < 0)
      unsqueezeAxis += outputShapeLen;
    unsqueezeAxesSet.insert(unsqueezeAxis);
  }

  // Compute the unsqueezed symbolic shape.
  SmallVector<TypedAttr> unsqueezedDims;
  TypedAttr intOneAttr = b.createVar(1).getAttr();
  for (int64_t idx = 0, nextDimAttrIdx = 0; idx < outputShapeLen; idx++) {
    if (unsqueezeAxesSet.contains(idx))
      unsqueezedDims.push_back(intOneAttr);
    else
      unsqueezedDims.push_back(inputShapeAttr[nextDimAttrIdx++]);
  }
  auto unsqueezedShapeAttr =
      ShapeAttr::get(unsqueezedDims, inputShapeAttr.getType());

  return unsqueezedShapeAttr;
}

} // namespace M::MO
