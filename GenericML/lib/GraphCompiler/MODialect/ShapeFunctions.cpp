//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/ShapeFunctions.h"

#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/ParserUtils.h"
#include <optional>
#include <utility>

namespace M::MO {

KGEN::ParamDeclareOp ParamBinding::toOp(Location loc, OpBuilder b) const {
  return b.create<KGEN::ParamDeclareOp>(loc, decl, value);
}

Var ParamExprBuilder::createVar(TypedAttr attr) {
  // No need to create a new binding if `attr` is simple.
  TypedAttr varAttr = wrapNonTrivialAttrWithDecl(attr, "var");
  return Var(varAttr, this);
}

Var ParamExprBuilder::createVar(int64_t n) {
  auto indexTy = IndexType::get(graphOp.getContext());
  return createVar(IntegerAttr::get(indexTy, n));
}

TypedAttr ParamExprBuilder::createKgenExpr(KGEN::POC poc,
                                           ArrayRef<TypedAttr> operands,
                                           StringRef declPrefix) {
  auto exprAttr = KGEN::ParamOperatorAttr::get(poc, operands);
  return wrapNonTrivialAttrWithDecl(exprAttr, declPrefix);
}

TypedAttr ParamExprBuilder::wrapNonTrivialAttrWithDecl(TypedAttr attr,
                                                       StringRef declPrefix) {
  if (isa<IntegerAttr, KGEN::ParamDeclRefAttr>(attr))
    return attr;

  Type resTy = attr.getType();
  KGEN::ParamDeclAttr declAttr = graphOp.getNewParamOfType(declPrefix, resTy);
  auto declRefAttr = KGEN::ParamDeclRefAttr::get(declAttr);
  createdOps->push_back(ParamBinding(declAttr, attr));
  return declRefAttr;
}

Var ParamExprBuilder::createCond(const Var &condV, const Var &thenV,
                                 const Var &elseV) {
  auto exprAttr = KGEN::ParamOperatorAttr::get(
      KGEN::POC::Cond, {condV.getAttr(), thenV.getAttr(), elseV.getAttr()});
  return createVar(wrapNonTrivialAttrWithDecl(exprAttr, "cond"));
}

Var ParamExprBuilder::createMax(const Var &lhs, const Var &rhs) {
  auto exprAttr = KGEN::ParamOperatorAttr::get(KGEN::POC::Max, lhs.getAttr(),
                                               rhs.getAttr());
  return createVar(wrapNonTrivialAttrWithDecl(exprAttr, "max"));
}

Var ParamExprBuilder::createMin(const Var &lhs, const Var &rhs) {
  auto exprAttr = KGEN::ParamOperatorAttr::get(KGEN::POC::Min, lhs.getAttr(),
                                               rhs.getAttr());
  return createVar(wrapNonTrivialAttrWithDecl(exprAttr, "min"));
}

ShapeAttr ParamExprBuilder::getShapeAttr(ArrayRef<TypedAttr> dimAttrs) {
  return ShapeAttr::get(dimAttrs, ShapeType::get(graphOp.getContext()));
}

void ParamExprBuilder::createAssert(const Var &condVar, StringRef errorMsg) {
  TypedAttr condAttr = condVar.getAttr();
  MLIRContext *ctx = condAttr.getContext();
  Type chainType = MO::ChainType::get(ctx);

  // Build a new operand list which starts with MLIROpAttr.
  auto signature = SignatureType::get(ctx, {condAttr.getType()}, chainType);
  NamedAttribute msgAttr(StringAttr::get(ctx, "message"),
                         StringAttr::get(errorMsg, KGEN::StringType::get(ctx)));
  auto dictAttr = DictionaryAttr::get(ctx, {msgAttr});
  auto opNameAttr = StringAttr::get(ctx, MO::AssertOp::getOperationName());
  SmallVector<TypedAttr> applyOperands{
      KGEN::MLIROpAttr::get(opNameAttr, dictAttr, signature), condAttr};

  (void)createKgenExpr(KGEN::POC::Apply, applyOperands, "assert");
}

/// Try converting `attr` to a constant integer.
std::optional<int64_t> tryGetConstantInt(TypedAttr attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getInt();
  return std::nullopt;
}

std::optional<SmallVector<int64_t>> tryGetConstantIntVector(TypedAttr attr) {
  // First check `attr` is an array of attrs.
  auto shapeAttr = dyn_cast<ShapeAttr>(attr);
  if (!shapeAttr)
    return std::nullopt;

  // Check each element attribute in the array, collect the constant integers,
  // or short-circuit if any attribute cannot be converted to a constant
  // integer.
  SmallVector<int64_t> result;
  for (TypedAttr maybeIntAttr : shapeAttr.getValues()) {
    if (auto intOr = tryGetConstantInt(maybeIntAttr))
      result.push_back(intOr.value());
    else
      return std::nullopt;
  }
  return result;
}

/// Compute the output spatial dimension for a sliding window op (e.g.,
/// convolution or pooling).
static Var getSlidingWindowOutDim(const Var &inputDim, const Var &filterDim,
                                  const Var &dilation, const Var &stride,
                                  const Var &padding, bool ceilMode = false) {
  Var numerator = inputDim + padding - (dilation * (filterDim - 1) + 1);
  if (ceilMode)
    return (numerator + stride - 1) / stride + 1;
  return numerator / stride + 1;
}

// Compute the size of a range (assuming validity of input values).
static Var computeRangeSize(const Var &start, const Var &stop,
                            const Var &step) {
  // We need to be careful with offset and negatives, think range(0, 3, 2) and
  // range(0, -3, -2);
  ParamExprBuilder &b = start.getBuilder();
  Var diff = stop - start;
  Var offset = b.createCond(step > 0, step - 1, step + 1);
  Var size = (diff + offset) / step;
  return size;
}

// Generate an error message when dimensions of a shape are
// not equivalent or broadcastable. This is meant for use with RMO ops.
static std::string
shapeBroadcastingErrorMessage(StringRef lhsName, ShapeAttr lhsShapeAttr,
                              size_t lhsAxis, StringRef rhsName,
                              ShapeAttr rhsShapeAttr, size_t rhsAxis) {
  std::string errorMsg;
  llvm::raw_string_ostream os(errorMsg);
  StreamAsmPrinter printer(os);
  printer << "Input " << lhsName << " (shape ";
  MOSH::printShapeTypedAttr(printer, lhsShapeAttr);
  printer << ") dimension at axis " << lhsAxis << " (value ";
  MOSH::printDimensionValue(printer, lhsShapeAttr[lhsAxis]);
  printer << ") and input " << rhsName << " (shape ";
  MOSH::printShapeTypedAttr(printer, rhsShapeAttr);
  printer << ") dimension at axis " << rhsAxis << " (value ";
  MOSH::printDimensionValue(printer, rhsShapeAttr[rhsAxis]);
  printer << ") are neither equivalent nor broadcastable. Insert an "
             "explicit rebind if these dims will be equivalent at runtime.";
  return errorMsg;
}

static std::string matmulInnerDimErrorMessage(ShapeAttr lhsShapeAttr,
                                              TypedAttr lhsDim,
                                              ShapeAttr rhsShapeAttr,
                                              TypedAttr rhsDim) {
  std::string errorMsg;
  llvm::raw_string_ostream os(errorMsg);
  StreamAsmPrinter printer(os);
  printer << "Matrix multiplication input lhs (shape ";
  MOSH::printShapeTypedAttr(printer, lhsShapeAttr);
  printer << ") dimension at axis -1 (value ";
  MOSH::printDimensionValue(printer, lhsDim);
  printer << ") must match input rhs (shape ";
  MOSH::printShapeTypedAttr(printer, rhsShapeAttr);
  printer << ") dimension at axis ";

  if (rhsShapeAttr.getRank() == 1)
    printer << "-1";
  else
    printer << "-2";

  printer << " (value ";
  MOSH::printDimensionValue(printer, rhsDim);
  printer << ")";
  return errorMsg;
}

// Calculates the full rank shapes for matmul inputs.
// These shapes should be equivalent after broadcasting.
// The new dimensions will just be static 1 dimensions.
std::pair<SmallVector<TypedAttr>, SmallVector<TypedAttr>>
expandMatmulInputsForBroadcasting(ShapeAttr lhsShape, ShapeAttr rhsShape) {
  size_t lhsRank = lhsShape.getRank();
  size_t rhsRank = rhsShape.getRank();

  size_t fullRank = std::max(size_t(2), std::max(lhsRank, rhsRank));

  auto idxTy = IndexType::get(lhsShape.getContext());
  auto intDim1 = IntegerAttr::get(idxTy, 1ll);
  SmallVector<TypedAttr> lhsFullShape(fullRank, intDim1);
  SmallVector<TypedAttr> rhsFullShape(fullRank, intDim1);
  if (lhsRank == 1) {
    // `D` -> `1xD`
    lhsFullShape[fullRank - 1] = lhsShape[0];
  } else {
    size_t offset = fullRank - lhsRank;
    for (size_t i = 0; i < lhsRank; ++i)
      lhsFullShape[i + offset] = lhsShape[i];
  }

  if (rhsRank == 1) {
    // `D` -> `Dx1`
    rhsFullShape[fullRank - 2] = rhsShape[0];
  } else {
    size_t offset = fullRank - rhsRank;
    for (size_t i = 0; i < rhsRank; ++i)
      rhsFullShape[i + offset] = rhsShape[i];
  }

  return {lhsFullShape, rhsFullShape};
}

//----------------------------------------------------------------------------//
//------------------------Shape functions begin-------------------------------//

// NOTE these helpers will eventually be shared between shape inference and
// GraphAPI shape propagation.
//
// These helpers have pretty uniform input types, where
// - `GraphOp` is used for new parameter name generation
// - `Location` is used for newly created instructions
// - All the `TypedAttr`s are parameteric representation of the op's input

/// Materialize the shape logic for `MO::BatchMatmulOp`, assuming given
ShapeMaterializeResult materializeBatchMatmulShape(GraphOp graphOp,
                                                   Location loc,
                                                   TypedAttr lhsShapeAttr,
                                                   TypedAttr rhsShapeAttr) {
  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // Make sure both input and target shapes are ranked symbolic shapes.
  ShapeAttr lhsShape = dyn_cast<ShapeAttr>(lhsShapeAttr);
  ShapeAttr rhsShape = dyn_cast<ShapeAttr>(rhsShapeAttr);
  if (!lhsShape || !rhsShape)
    return ShapeMaterializeResult::nonMaterializable();

  size_t inputRank = lhsShape.getRank();
  if (rhsShape.getRank() != inputRank || inputRank <= 2) {
    return ShapeMaterializeResult::error(
        "[batch_matmul] input ranks must match and exceed 2");
  }

  SmallVector<TypedAttr> resDims;
  // Check batch dimensions are equal and collect them into `resDims`.
  for (size_t batchDim = 0; batchDim < inputRank - 2; batchDim++) {
    Var lhsDim = b.createVar(lhsShape[batchDim]);
    Var rhsDim = b.createVar(rhsShape[batchDim]);
    b.createAssert(lhsDim == rhsDim,
                   "[batch_matmul] inputs batch dimensions must match");
    resDims.push_back(lhsDim.getAttr());
  }

  // Check inner dimensions are equal
  Var lhsInnerDim = b.createVar(lhsShape[inputRank - 1]);
  Var rhsInnerDim = b.createVar(rhsShape[inputRank - 2]);
  b.createAssert(lhsInnerDim == rhsInnerDim,
                 "[batch_matmul] inputs inner dimensions must match");

  // Collect outer dims.
  resDims.push_back(lhsShape[inputRank - 2]);
  resDims.push_back(rhsShape[inputRank - 1]);
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

ShapeMaterializeResult materializeRmoMatmulShape(GraphOp graphOp, Location loc,
                                                 TypedAttr lhsShapeAttr,
                                                 TypedAttr rhsShapeAttr) {
  // Create builder and track generated parameter bindings.
  mlir::MLIRContext *ctx = lhsShapeAttr.getContext();
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // Make sure both input and target shapes are ranked symbolic shapes.
  ShapeAttr lhsShape = dyn_cast<ShapeAttr>(lhsShapeAttr);
  ShapeAttr rhsShape = dyn_cast<ShapeAttr>(rhsShapeAttr);
  if (!lhsShape || !rhsShape)
    return ShapeMaterializeResult::nonMaterializable();

  // Note: this supports both vector and matrix matmuls.
  // Fundamentally, vectors are consider a `1xD` or `Dx1` matrix.
  // That said, the resulting `1` dimensions in the output is squeezed away.
  // All dimensions outside of the innermost 2 are broadcasted.
  size_t lhsRank = lhsShape.getRank();
  size_t rhsRank = rhsShape.getRank();
  if (lhsRank == 0 || rhsRank == 0)
    return ShapeMaterializeResult::error("Scalar inputs are not supported");

  // The full shape/rank has all of the dimensions that the underlying
  // mo.matmul/mo.batch_matmul will execute with.
  // The new dimensions will just be static 1 dimensions.
  size_t fullRank = std::max(size_t(2), std::max(lhsRank, rhsRank));

  auto idxTy = IndexType::get(ctx);
  auto intDim1 = IntegerAttr::get(idxTy, 1ll);
  auto [lhsFullShape, rhsFullShape] =
      expandMatmulInputsForBroadcasting(lhsShape, rhsShape);

  // Check inner dimensions are equal
  TypedAttr lhsInnerDim = lhsFullShape[fullRank - 1];
  TypedAttr rhsInnerDim = rhsFullShape[fullRank - 2];
  std::string errorMsg =
      matmulInnerDimErrorMessage(lhsShape, lhsInnerDim, rhsShape, rhsInnerDim);
  b.createAssert(b.createVar(lhsInnerDim) == b.createVar(rhsInnerDim),
                 errorMsg);

  SmallVector<TypedAttr> resDims;
  resDims.reserve(fullRank);
  // Check batch dimensions are equal or broadcastable and collect them into
  // `resDims`.
  size_t lhsRankOffset = fullRank - lhsRank;
  size_t rhsRankOffset = fullRank - rhsRank;
  for (size_t batchDim = 0; batchDim < fullRank - 2; batchDim++) {
    if (lhsFullShape[batchDim] == intDim1) {
      // lhs broadcastable, pick rhs dim as output.
      resDims.push_back(rhsFullShape[batchDim]);
    } else if (rhsFullShape[batchDim] == intDim1) {
      // rhs broadcastable, pick lhs dim as output.
      resDims.push_back(lhsFullShape[batchDim]);
    } else {
      // Neither broadcastable, must be equal.
      TypedAttr lhsDim = lhsFullShape[batchDim];
      TypedAttr rhsDim = rhsFullShape[batchDim];

      std::string errorMsg = shapeBroadcastingErrorMessage(
          "lhs", lhsShape, batchDim - lhsRankOffset, "rhs", rhsShape,
          batchDim - rhsRankOffset);
      b.createAssert(b.createVar(lhsDim) == b.createVar(rhsDim), errorMsg);
      resDims.push_back(lhsDim);
    }
  }

  // Collect inner dims if inputs were not 1d.
  if (lhsRank != 1)
    resDims.push_back(lhsFullShape[fullRank - 2]);
  if (rhsRank != 1)
    resDims.push_back(rhsFullShape[fullRank - 1]);

  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

ShapeMaterializeResult
materializeRmoBinaryElementwiseShape(GraphOp graphOp, Location loc,
                                     ShapeAttr lhsShape, ShapeAttr rhsShape) {
  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  auto idxTy = IndexType::get(graphOp.getContext());
  auto intDim1 = IntegerAttr::get(idxTy, 1ll);
  size_t maxRank = std::max(lhsShape.getRank(), rhsShape.getRank());

  auto getFromEnd = [](ShapeAttr shape, size_t i) -> TypedAttr {
    if (i >= shape.getRank())
      return nullptr;
    return shape[shape.getRank() - i - 1];
  };

  // Check that all inputs are equivalent or broadcastable while building the
  // output shape.
  SmallVector<TypedAttr> outputDims{maxRank, intDim1};
  for (size_t i = 0; i < maxRank; ++i) {
    TypedAttr lhsDim = getFromEnd(lhsShape, i);
    TypedAttr rhsDim = getFromEnd(rhsShape, i);
    size_t outputIndex = maxRank - i - 1;
    if (lhsDim && rhsDim) {
      if (lhsDim == intDim1) {
        outputDims[outputIndex] = rhsDim;
      } else if (rhsDim == intDim1) {
        outputDims[outputIndex] = lhsDim;
      } else {
        b.createAssert(b.createVar(lhsDim) == b.createVar(rhsDim),
                       shapeBroadcastingErrorMessage(
                           "lhs", lhsShape, lhsShape.getRank() - i - 1, "rhs",
                           rhsShape, rhsShape.getRank() - i - 1));
        outputDims[outputIndex] = lhsDim;
      }
    } else if (lhsDim) {
      outputDims[outputIndex] = lhsDim;
    } else {
      outputDims[outputIndex] = rhsDim;
    }
  }

  auto outputShape =
      ShapeAttr::get(outputDims, ShapeType::get(graphOp.getContext()));
  return ShapeMaterializeResult::success(std::move(paramBindings), outputShape);
}

ShapeMaterializeResult materializeRmoSelectShape(GraphOp graphOp, Location loc,
                                                 ShapeAttr condShape,
                                                 ShapeAttr xShape,
                                                 ShapeAttr yShape) {
  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  auto idxTy = IndexType::get(graphOp.getContext());
  auto intDim1 = IntegerAttr::get(idxTy, 1ll);
  size_t maxRank = std::max(condShape.getRank(),
                            std::max(xShape.getRank(), yShape.getRank()));

  auto getFromEndOrOne = [intDim1](ShapeAttr shape, size_t i) -> TypedAttr {
    if (i >= shape.getRank())
      return intDim1;
    return shape[shape.getRank() - i - 1];
  };

  // Check that all inputs are equivalent or broadcastable while building the
  // output shape.
  SmallVector<TypedAttr> outputDims{maxRank, intDim1};
  for (size_t i = 0; i < maxRank; ++i) {
    TypedAttr condDim = getFromEndOrOne(condShape, i);
    TypedAttr xDim = getFromEndOrOne(xShape, i);
    TypedAttr yDim = getFromEndOrOne(yShape, i);
    size_t outputIndex = maxRank - i - 1;

    SmallVector<std::tuple<StringRef, ShapeAttr, TypedAttr>, 3> nonOneDims{};
    if (condDim != intDim1)
      nonOneDims.push_back({"condition", condShape, condDim});
    if (xDim != intDim1)
      nonOneDims.push_back({"x", xShape, xDim});
    if (yDim != intDim1)
      nonOneDims.push_back({"y", yShape, yDim});

    for (size_t j = 1; j < nonOneDims.size(); ++j) {
      auto [expectedName, expectedShape, expectedDim] = nonOneDims[0];
      auto [actualName, actualShape, actualDim] = nonOneDims[j];
      b.createAssert(b.createVar(expectedDim) == b.createVar(actualDim),
                     shapeBroadcastingErrorMessage(
                         expectedName, expectedShape,
                         expectedShape.getRank() - i - 1, actualName,
                         actualShape, actualShape.getRank() - i - 1));
    }

    if (!nonOneDims.empty()) {
      auto [_name, _shape, dim] = nonOneDims[0];
      outputDims[outputIndex] = dim;
    }
  }

  auto outputShape =
      ShapeAttr::get(outputDims, ShapeType::get(graphOp.getContext()));
  return ShapeMaterializeResult::success(std::move(paramBindings), outputShape);
}

ShapeMaterializeResult materializeBroadcastShapeShape(GraphOp graphOp,
                                                      Location loc,
                                                      TypedAttr lhsShapeAttr,
                                                      TypedAttr rhsShapeAttr) {
  // Make sure both lhs and rhs shapes are ranked.
  ShapeAttr lhsShape = dyn_cast<ShapeAttr>(lhsShapeAttr);
  ShapeAttr rhsShape = dyn_cast<ShapeAttr>(rhsShapeAttr);
  if (!lhsShape || !rhsShape)
    return ShapeMaterializeResult::nonMaterializable();

  // Simple rank check.
  if (lhsShape.getRank() != 1 || rhsShape.getRank() != 1) {
    return ShapeMaterializeResult::error(
        "[broadcast_shape] inputs must be rank 1 shape tensors");
  }

  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);
  Var lhsDim = b.createVar(lhsShape[0]);
  Var rhsDim = b.createVar(rhsShape[0]);
  ShapeAttr outShape = b.getShapeAttr(b.createMax(lhsDim, rhsDim));
  return ShapeMaterializeResult::success(std::move(paramBindings), outShape);
}

/// Materialize the shape logic for `MO::BroadcastToOp`.
ShapeMaterializeResult materializeBroadcastToShape(GraphOp graphOp,
                                                   Location loc,
                                                   TypedAttr inputShapeAttr,
                                                   TypedAttr targetShapeAttr) {
  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // Make sure both input and target shapes are ranked symbolic shapes.
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr targetShape = dyn_cast<ShapeAttr>(targetShapeAttr);
  if (!inputShape || !targetShape)
    return ShapeMaterializeResult::nonMaterializable();

  int inputRank = inputShape.getRank();
  int outputRank = targetShape.getRank();
  if (inputRank > outputRank) {
    return ShapeMaterializeResult::error("[broadcast_to] input ranks must less "
                                         "than or equal to the output rank");
  }

  for (int i = 0; i < inputRank; i++) {
    int inputAxis = inputRank - i - 1;
    int outputAxis = outputRank - i - 1;
    Var inputDim = b.createVar(inputShape[inputAxis]);
    Var outputDim = b.createVar(targetShape[outputAxis]);

    // Insert assertions
    b.createAssert(inputDim == 1 || inputDim == outputDim,
                   "[broadcast_to] input dim must be either 1 or equal to "
                   "corresponding output dim starting from the rightmost dim");
  }
  return ShapeMaterializeResult::success(std::move(paramBindings), targetShape);
}

/// Materialize the shape logic for `MO::ConcatOp`.
ShapeMaterializeResult
materializeConcatShape(GraphOp graphOp, Location loc, TypedAttr axisAttr,
                       ArrayRef<TypedAttr> inputShapeAttrs) {
  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // Check for constant axis
  std::optional<int64_t> axisOr = tryGetConstantInt(axisAttr);
  if (!axisOr.has_value())
    return ShapeMaterializeResult::nonMaterializable();
  int64_t axis = axisOr.value();

  // Check for statically-ranked input shapes.
  SmallVector<ShapeAttr> inputShapes;
  for (TypedAttr inputShapeAttr : inputShapeAttrs) {
    if (auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr)) {
      inputShapes.push_back(inputShape);
    } else {
      // Give up on any dyn-ranked input shape
      return ShapeMaterializeResult::nonMaterializable();
    }
  }

  // `MO::ConcatOp` verifier ensures there's at least 1 input, and ranks of
  // inputs all match.
  ShapeAttr firstInputShape = inputShapes.front();
  auto restInputShapes = ArrayRef(inputShapes).drop_front();
  int64_t inputRank = firstInputShape.getRank();

  // Normalize axis and make sure it's in bound.
  axis = axis < 0 ? axis + inputRank : axis;
  if (axis < 0 || inputRank <= axis) {
    return ShapeMaterializeResult::error(
        "[concat] normalized axis must be within range [0, input_rank)");
  }

  // Construct output shape and make sure input shapes are compatible.
  SmallVector<TypedAttr> resDims(firstInputShape.getValues());
  Var concatDimSum = b.createVar(resDims[axis]);
  for (ShapeAttr inputShape : restInputShapes) {
    for (auto [idx, firstInputDimAttr, inputDimAttr] :
         llvm::enumerate(firstInputShape.getValues(), inputShape.getValues())) {
      if (static_cast<int64_t>(idx) != axis) {
        Var firstInputDim = b.createVar(firstInputDimAttr);
        Var inputDim = b.createVar(inputDimAttr);
        b.createAssert(
            firstInputDim == inputDim,
            "[concat] input shapes must match except at concat axis");
      }
    }
    Var inputAxisDim = b.createVar(inputShape[axis]);
    concatDimSum = concatDimSum + inputAxisDim;
  }
  resDims[axis] = concatDimSum.getAttr();
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

ShapeMaterializeResult
materializeConvShape(GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
                     TypedAttr filterShapeAttr, TypedAttr stridesAttr,
                     TypedAttr dilationsAttr, TypedAttr paddingsAttr,
                     TypedAttr numGroupsAttr) {
  auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  auto filterShape = dyn_cast<ShapeAttr>(filterShapeAttr);
  auto strides = dyn_cast<ShapeAttr>(stridesAttr);
  auto dilations = dyn_cast<ShapeAttr>(dilationsAttr);
  auto paddings = dyn_cast<ShapeAttr>(paddingsAttr);
  // Make sure both input and target shapes are ranked symbolic shapes.
  // No need to check for `numGroupsAttr` because it's just an integer.
  if (!inputShape || !filterShape || !strides || !dilations || !paddings) {
    return ShapeMaterializeResult::nonMaterializable();
  }

  // Preliminary rank checks.
  size_t inputRank = inputShape.getRank();
  if (inputRank < 3) {
    return ShapeMaterializeResult::error(
        "[convolution] requires (input_rank >= 3)");
  }
  if (inputRank != filterShape.getRank()) {
    return ShapeMaterializeResult::error(
        "[convolution] requires (input_rank == filter_rank)");
  }
  if (strides.getSize() != inputRank - 2 ||
      dilations.getSize() != inputRank - 2) {
    return ShapeMaterializeResult::error(
        "[convolution] requires (len(strides) == len(dilations) == input_rank "
        "- 2)");
  }
  if (paddings.getSize() != 2 * (inputRank - 2)) {
    return ShapeMaterializeResult::error(
        "[convolution] requires (len(paddings) == 2 * (input rank - 2))");
  }

  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // NOTE
  // - input and output have layout [batch_size, ...spatial_dims...,
  //   input_channels]
  // - filter has layout [...spatial_dims..., filter_channels, output_channels]
  Var batchSize = b.createVar(inputShape[0]);
  Var inputChannels = b.createVar(inputShape[inputRank - 1]);
  Var filterChannels = b.createVar(filterShape[inputRank - 2]);
  Var outputChannels = b.createVar(filterShape[inputRank - 1]);
  Var numGroups = b.createVar(numGroupsAttr);

  // More checks on channels and numGroups.
  b.createAssert(inputChannels == (numGroups * filterChannels),
                 "[convolution] requires (input_channels == num_groups * "
                 "filter_channels)");
  b.createAssert(
      (outputChannels % numGroups) == 0,
      "[convolution] output_channels must be divisible by num_groups");

  // Compute and collect output spatial dimensions.
  SmallVector<TypedAttr> resDims;
  resDims.push_back(batchSize.getAttr());
  for (size_t axis = 0; axis < inputRank - 2; axis++) {
    Var inputSpatialDim = b.createVar(inputShape[axis + 1]);
    Var filterSpatialDim = b.createVar(filterShape[axis]);
    Var dilation = b.createVar(dilations[axis]);
    Var stride = b.createVar(strides[axis]);
    Var prePadding = b.createVar(paddings[2 * axis]);
    Var postPadding = b.createVar(paddings[2 * axis + 1]);

    Var outputSpatialDim =
        getSlidingWindowOutDim(inputSpatialDim, filterSpatialDim, dilation,
                               stride, prePadding + postPadding);
    b.createAssert(outputSpatialDim > 0,
                   "[convolution] output spatial dim must be positive");
    resDims.push_back(outputSpatialDim.getAttr());
  }

  // Create output shape
  resDims.push_back(outputChannels.getAttr());
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

ShapeMaterializeResult materializeConvTransposeShape(
    GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
    TypedAttr filterShapeAttr, TypedAttr stridesAttr, TypedAttr dilationsAttr,
    TypedAttr paddingsAttr, TypedAttr outputPaddingsAttr) {
  auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  auto filterShape = dyn_cast<ShapeAttr>(filterShapeAttr);
  auto strides = dyn_cast<ShapeAttr>(stridesAttr);
  auto dilations = dyn_cast<ShapeAttr>(dilationsAttr);
  auto paddings = dyn_cast<ShapeAttr>(paddingsAttr);
  auto outPaddings = dyn_cast<ShapeAttr>(outputPaddingsAttr);
  // Make sure both input and target shapes are ranked symbolic shapes.
  // No need to check for `numGroupsAttr` because it's just an integer.
  if (!inputShape || !filterShape || !strides || !dilations || !paddings ||
      !outPaddings) {
    return ShapeMaterializeResult::nonMaterializable();
  }

  // Preliminary rank checks.
  size_t inputRank = inputShape.getRank();
  if (inputRank != 4 && inputRank != 5) {
    return ShapeMaterializeResult::error(
        "[conv_transpose] requires (input_rank == 4 or 5)");
  }
  if (inputRank != filterShape.getRank()) {
    return ShapeMaterializeResult::error(
        "[conv_transpose] requires (input_rank == kernel_rank)");
  }
  if (strides.getSize() != inputRank - 2 ||
      dilations.getSize() != inputRank - 2) {
    return ShapeMaterializeResult::error(
        "[conv_transpose] requires (len(strides) == len(dilations) == "
        "input_rank - 2)");
  }
  if (paddings.getSize() != 2 * (inputRank - 2)) {
    return ShapeMaterializeResult::error(
        "[conv_transpose] requires (len(paddings) == 2 * (input rank - 2))");
  }

  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // NOTE
  // - input and output have layout [batch_size, ...spatial_dims...,
  // input_channels]
  // - filter has layout [...spatial_dims..., filter_channels, output_channels]
  //
  // BUT since ConvTranspose is the inverse of Conv w.r.t. the input, the output
  // shape would have layout [batch_size, ...spatial_dims..., filter_channels]
  Var batchSize = b.createVar(inputShape[0]);
  Var inputChannels = b.createVar(inputShape[inputRank - 1]);
  Var filterChannels = b.createVar(filterShape[inputRank - 2]);
  Var outputChannels = b.createVar(filterShape[inputRank - 1]);

  // Check input and kernel channels.
  b.createAssert(inputChannels == outputChannels,
                 "[conv_transpose] requires input channel to match output "
                 "channel in kernel");

  SmallVector<TypedAttr> resDims;
  resDims.push_back(batchSize.getAttr());

  // Compute a specific output spatial dimension.
  auto computeOutDim = [](const Var &inputDim, const Var &filterDim,
                          const Var &dilation, const Var &stride,
                          const Var &prePad, const Var &postPad,
                          const Var &outPad) -> Var {
    return stride * (inputDim - 1) + outPad + ((filterDim - 1) * dilation + 1) -
           prePad - postPad;
  };

  // Compute and collect output spatial dimensions;
  for (size_t axis = 0; axis < inputRank - 2; axis++) {
    Var inputSpatialDim = b.createVar(inputShape[axis + 1]);
    Var filterSpatialDim = b.createVar(filterShape[axis]);
    Var dilation = b.createVar(dilations[axis]);
    Var stride = b.createVar(strides[axis]);
    Var prePad = b.createVar(paddings[2 * axis]);
    Var postPad = b.createVar(paddings[2 * axis + 1]);
    Var outPad = b.createVar(outPaddings[axis]);

    Var outputSpatialDim =
        computeOutDim(inputSpatialDim, filterSpatialDim, dilation, stride,
                      prePad, postPad, outPad);
    b.createAssert(outputSpatialDim > 0,
                   "[conv_transpose] output spatial dim must be positive");
    resDims.push_back(outputSpatialDim.getAttr());
  }

  // Create output shape
  resDims.push_back(filterChannels.getAttr());
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

/// Materialize the shape logic for `MO::BatchMatmulOp`, assuming given
ShapeMaterializeResult materializeMatrixSolveShape(GraphOp graphOp,
                                                   Location loc,
                                                   TypedAttr lhsShapeAttr,
                                                   TypedAttr rhsShapeAttr) {
  // Make sure both input and target shapes are ranked symbolic shapes.
  ShapeAttr lhsShape = dyn_cast<ShapeAttr>(lhsShapeAttr);
  ShapeAttr rhsShape = dyn_cast<ShapeAttr>(rhsShapeAttr);
  if (!lhsShape || !rhsShape)
    return ShapeMaterializeResult::nonMaterializable();

  // Basic rank check.
  size_t inputRank = lhsShape.getRank();
  if (rhsShape.getRank() != inputRank) {
    return ShapeMaterializeResult::error(
        "[matrix-solve] requires input ranks to match");
  }
  if (inputRank < 2)
    return ShapeMaterializeResult::error("[matrix-solve] requires (rank >= 2)");

  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // lhsShape = [..., lhsOuterDim, lhsInnerDim]
  Var lhsOuterDim = b.createVar(lhsShape[inputRank - 2]);
  Var lhsInnerDim = b.createVar(lhsShape[inputRank - 1]);
  b.createAssert(
      lhsOuterDim == lhsInnerDim,
      "[matrix-solve] first input must be a (batch of) square matrix");

  // rhsShape = [..., rhsInnerDim, _]
  Var rhsInnerDim = b.createVar(rhsShape[inputRank - 2]);
  b.createAssert(lhsOuterDim == rhsInnerDim,
                 "[matrix-solve] inputs outter dimensions must match");

  // Check batch dims.
  for (size_t batchDim = 0; batchDim < inputRank - 2; batchDim++) {
    Var lhsDim = b.createVar(lhsShape[batchDim]);
    Var rhsDim = b.createVar(rhsShape[batchDim]);
    b.createAssert(lhsDim == rhsDim,
                   "[matrix-solve] inputs batch dimensions must match");
  }

  return ShapeMaterializeResult::success(std::move(paramBindings), rhsShape);
}

ShapeMaterializeResult materializePadShape(GraphOp graphOp, Location loc,
                                           TypedAttr inputShapeAttr,
                                           TypedAttr paddingsAttr) {
  auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  auto paddings = dyn_cast<ShapeAttr>(paddingsAttr);
  // Make sure input shape is ranked and paddings is a list of symbolic integers
  // with known length.
  if (!inputShape || !paddings) {
    return ShapeMaterializeResult::nonMaterializable();
  }

  // Preliminary rank checks.
  size_t inputRank = inputShape.getRank();
  if (2 * inputRank != paddings.getSize()) {
    return ShapeMaterializeResult::error(
        "[pad] paddings shape must be (2 * input_rank)");
  }

  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // Compute and collect output dimensions.
  SmallVector<TypedAttr> resDims;
  for (size_t axis = 0; axis < inputRank; axis++) {
    Var inputDim = b.createVar(inputShape[axis]);
    Var prePadding = b.createVar(paddings[2 * axis]);
    Var postPadding = b.createVar(paddings[2 * axis + 1]);
    Var outputDim = inputDim + prePadding + postPadding;
    resDims.push_back(outputDim.getAttr());
  }

  // Create output shape
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

ShapeMaterializeResult
materializePoolShape(GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
                     TypedAttr filterAttr, TypedAttr stridesAttr,
                     TypedAttr dilationsAttr, TypedAttr paddingsAttr,
                     bool ceilMode) {
  auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  auto filter = dyn_cast<ShapeAttr>(filterAttr);
  auto strides = dyn_cast<ShapeAttr>(stridesAttr);
  auto dilations = dyn_cast<ShapeAttr>(dilationsAttr);
  auto paddings = dyn_cast<ShapeAttr>(paddingsAttr);
  // Make sure both input and target shapes are ranked symbolic shapes.
  if (!inputShape || !filter || !strides || !dilations || !paddings) {
    return ShapeMaterializeResult::nonMaterializable();
  }

  // Preliminary rank checks.
  size_t inputRank = inputShape.getRank();
  // NOTE it's a trivial extension to enable this on arbitrary ranks, but I'm
  // keeping it the same as the mojo shape function for now.
  if (inputRank != 4) {
    return ShapeMaterializeResult::error(
        "[pooling] requires (input_rank == 4)");
  }

  size_t filterSize = filter.getSize();
  if (filterSize != inputRank - 2 ||
      !llvm::all_equal({filterSize, dilations.getSize(), strides.getSize()})) {
    return ShapeMaterializeResult::error(
        "[pooling] requires (len(filter) == len(strides) == len(dilations) == "
        "input_rank - 2)");
  }
  if (paddings.getSize() != 2 * (inputRank - 2)) {
    return ShapeMaterializeResult::error(
        "[pooling] requires (len(paddings) == 2 * (input rank - 2))");
  }

  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // NOTE
  // - input and output have layout [batch_size, ...spatial_dims...,
  // input_channels]
  // - filter has layout [...spatial_dims..., filter_channels, output_channels]
  Var batchSize = b.createVar(inputShape[0]);
  Var inputChannels = b.createVar(inputShape[inputRank - 1]);

  // Compute and collect output spatial dimensions.
  SmallVector<TypedAttr> resDims;
  resDims.push_back(batchSize.getAttr());
  for (size_t axis = 0; axis < inputRank - 2; axis++) {
    Var inputSpatialDim = b.createVar(inputShape[axis + 1]);
    Var filterSpatialDim = b.createVar(filter[axis]);
    Var dilation = b.createVar(dilations[axis]);
    Var stride = b.createVar(strides[axis]);
    Var prePadding = b.createVar(paddings[2 * axis]);
    Var postPadding = b.createVar(paddings[2 * axis + 1]);

    Var outputSpatialDim =
        getSlidingWindowOutDim(inputSpatialDim, filterSpatialDim, dilation,
                               stride, prePadding + postPadding, ceilMode);
    b.createAssert(outputSpatialDim > 0,
                   "[pooling] output spatial dim must be positive");
    resDims.push_back(outputSpatialDim.getAttr());
  }

  // Create output shape
  resDims.push_back(inputChannels.getAttr());
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

ShapeMaterializeResult materializeGatherShape(GraphOp graphOp, Location loc,
                                              TypedAttr inputShapeAttr,
                                              TypedAttr indicesShapeAttr,
                                              TypedAttr axisAttr) {
  // Make sure input and indices shapes are ranked, and axis is constant.
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr indicesShape = dyn_cast<ShapeAttr>(indicesShapeAttr);
  auto axisOr = tryGetConstantInt(axisAttr);
  if (!inputShape || !indicesShape || !axisOr.has_value())
    return ShapeMaterializeResult::nonMaterializable();

  // Normalize constant and make sure it's in bound.
  size_t inputRank = inputShape.getRank();
  int64_t axis = axisOr.value();
  axis = axis < 0 ? axis + inputRank : axis;
  if (axis < 0 || inputRank <= static_cast<size_t>(axis)) {
    return ShapeMaterializeResult::error(
        "[gather] normalized axis must be within range [0, input_rank)");
  }

  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // Compute and collect output dimensions.
  // Note the output shape is essentially replacing the `axis` dimension of
  // `input_shape` with `indices_shape`, i.e.,
  //   output_shape := input_shape[:axis] + indices_shape + input_shape[axis+1:]
  SmallVector<TypedAttr> resDims;
  // input_shape[:axis]
  for (int64_t idx = 0; idx < axis; idx++)
    resDims.push_back(inputShape[idx]);
  // indices_shape
  llvm::append_range(resDims, indicesShape.getValues());
  // input_shape[axis+1:]
  for (size_t idx = axis + 1; idx < inputRank; idx++)
    resDims.push_back(inputShape[idx]);

  // Create and return output shape.
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

ShapeMaterializeResult materializeGatherNdShape(GraphOp graphOp, Location loc,
                                                TypedAttr inputShapeAttr,
                                                TypedAttr indicesShapeAttr,
                                                int64_t batchDims) {
  // Make sure input and indices shapes are ranked.
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr indicesShape = dyn_cast<ShapeAttr>(indicesShapeAttr);
  if (!inputShape || !indicesShape)
    return ShapeMaterializeResult::nonMaterializable();

  // Preliminary rank checks
  size_t inputRank = inputShape.getRank();
  size_t indicesRank = indicesShape.getRank();
  if (inputRank < 1 || indicesRank < 1) {
    return ShapeMaterializeResult::error(
        "[gather_nd] input_rank and indices_rank must be >= 1");
  }

  // Size of the indexing dimension must be a constant
  auto idxSizeOr = tryGetConstantInt(indicesShape[indicesRank - 1]);
  if (!idxSizeOr.has_value())
    return ShapeMaterializeResult::nonMaterializable();
  int64_t idxSize = idxSizeOr.value();

  // More shape checks.
  if (idxSize < 1 || inputRank - batchDims < static_cast<size_t>(idxSize)) {
    return ShapeMaterializeResult::error(
        "[gather_nd] index size must be within range [1, input_rank - "
        "batch_dims]");
  }
  if (batchDims >= static_cast<int64_t>(indicesRank)) {
    return ShapeMaterializeResult::error(
        "[gather_nd] requires (batch_dims < indices_rank)");
  }

  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // Compute and collect output dimensions.
  //   output_shape = input_shape[:batch_dims]
  //                  + indices_shape[batch_dims:-1]
  //                  + input_shape[batch_dims + indices_shape[-1]:]
  SmallVector<TypedAttr> resDims;

  // input_shape[:batch_dims]
  for (int64_t idx = 0; idx < batchDims; idx++)
    resDims.push_back(inputShape[idx]);
  // indices_shape[batch_dims:-1]
  llvm::append_range(resDims,
                     indicesShape.getValues().slice(batchDims).drop_back());
  // input_shape[batch_dims + indices_shape[-1]:]
  for (size_t idx = batchDims + idxSize; idx < inputRank; idx++)
    resDims.push_back(inputShape[idx]);

  // Create and return output shape.
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

// Materialize the shape logic for operation's shape is stored as the *value*
// of the given input.
ShapeMaterializeResult materializeTargetShapeLike(GraphOp graphOp, Location loc,
                                                  TypedAttr targetShapeAttr) {
  auto asShapeAttr = dyn_cast<ShapeAttr>(targetShapeAttr);
  if (!asShapeAttr)
    return ShapeMaterializeResult::nonMaterializable();

  return ShapeMaterializeResult::success({}, asShapeAttr);
}

/// Materialize the shape logic for `MO::RangeOp`.
ShapeMaterializeResult materializeRangeShape(GraphOp graphOp, Location loc,
                                             TypedAttr startAttr,
                                             TypedAttr stopAttr,
                                             TypedAttr stepAttr) {
  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  Var start = b.createVar(startAttr);
  Var stop = b.createVar(stopAttr);
  Var step = b.createVar(stepAttr);

  // Check step != 0
  b.createAssert(step != 0, "[range] step must be non-zero");

  // Check start/stop are in bounds
  b.createAssert(start <= stop || step < 0,
                 "[range] requires (start <= stop) for positive step");
  b.createAssert(stop <= start || step > 0,
                 "[range] requires (stop <= start) for negative step");

  Var outDim = computeRangeSize(start, stop, step);
  ShapeAttr outputShape = b.getShapeAttr(outDim);
  return ShapeMaterializeResult::success(std::move(paramBindings), outputShape);
}

ShapeMaterializeResult materializeReduceShape(GraphOp graphOp, Location loc,
                                              TypedAttr inputShape,
                                              TypedAttr axis) {
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  auto axisOr = tryGetConstantInt(axis);
  auto inputShapeAttr = dyn_cast<ShapeAttr>(inputShape);

  if (!axisOr.has_value() || !inputShapeAttr)
    return ShapeMaterializeResult::nonMaterializable();

  int64_t rank = inputShapeAttr.getRank();
  int64_t normalizedAxis = (*axisOr < 0) ? (*axisOr + rank) : *axisOr;

  if (normalizedAxis < 0 || normalizedAxis >= rank)
    return ShapeMaterializeResult::error(
        "[reduction] normalized axis must be within range [0, input_rank)");

  SmallVector<TypedAttr> finalShape(inputShapeAttr.getValues());
  finalShape[normalizedAxis] = b.createVar(1).getAttr();
  return ShapeMaterializeResult::success(std::move(paramBindings),
                                         b.getShapeAttr(finalShape));
}

ShapeMaterializeResult materializeScatterShape(GraphOp graphOp, Location loc,
                                               TypedAttr inputShapeAttr,
                                               TypedAttr updatesShapeAttr,
                                               TypedAttr indicesShapeAttr,
                                               TypedAttr axisAttr) {
  // Make sure input, updates and indices shapes are ranked, and axis is
  // constant.
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr updatesShape = dyn_cast<ShapeAttr>(updatesShapeAttr);
  ShapeAttr indicesShape = dyn_cast<ShapeAttr>(indicesShapeAttr);
  auto axisOr = tryGetConstantInt(axisAttr);
  if (!inputShape || !updatesShape, !indicesShape || !axisOr.has_value())
    return ShapeMaterializeResult::nonMaterializable();

  // Basic rank check
  size_t inputRank = inputShape.getRank();
  if (!llvm::all_equal(
          {inputRank, updatesShape.getRank(), indicesShape.getRank()})) {
    return ShapeMaterializeResult::error(
        "[scatter] rank of inputs must be equal");
  }

  // Normalize constant and make sure it's in bound.
  int64_t axis = axisOr.value();
  axis = axis < 0 ? axis + inputRank : axis;
  if (axis < 0 || inputRank <= static_cast<size_t>(axis)) {
    return ShapeMaterializeResult::error(
        "[scatter] normalized axis must be within range [0, input_rank)");
  }

  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // Check dimensions of input shapes.
  for (size_t i = 0; i < inputRank; i++) {
    Var inputDim = b.createVar(inputShape[i]);
    Var updatesDim = b.createVar(updatesShape[i]);
    Var indicesDim = b.createVar(indicesShape[i]);
    b.createAssert(indicesDim == updatesDim,
                   "[scatter] indices and updates must have the same shape");
    b.createAssert(indicesDim <= inputDim,
                   "[scatter] indices shape cannot be bigger than input shape");
  }

  // Create and return output shape.
  return ShapeMaterializeResult::success(std::move(paramBindings), inputShape);
}

ShapeMaterializeResult materializeScatterNdShape(GraphOp graphOp, Location loc,
                                                 TypedAttr inputShapeAttr,
                                                 TypedAttr updatesShapeAttr,
                                                 TypedAttr indicesShapeAttr) {
  // Make sure input, updates and indices shapes are ranked, and axis is
  // constant.
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr updatesShape = dyn_cast<ShapeAttr>(updatesShapeAttr);
  ShapeAttr indicesShape = dyn_cast<ShapeAttr>(indicesShapeAttr);
  if (!inputShape || !updatesShape, !indicesShape)
    return ShapeMaterializeResult::nonMaterializable();

  // Basic rank check.
  int64_t inputRank = inputShape.getRank();
  int64_t updatesRank = updatesShape.getRank();
  int64_t indicesRank = indicesShape.getRank();
  if (indicesRank < 1) {
    return ShapeMaterializeResult::error(
        "[scatter_nd] indices cannot be a scalar");
  }

  // Number of sliced dimensions must be constant
  auto numSlicedDimsOr = tryGetConstantInt(indicesShape[indicesRank - 1]);
  if (!numSlicedDimsOr.has_value())
    return ShapeMaterializeResult::nonMaterializable();
  int64_t numSlicedDims = numSlicedDimsOr.value();

  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // More rank checks.
  if (numSlicedDims > inputRank) {
    return ShapeMaterializeResult::error(
        "[scatter_nd] cannot slice more dimensions than what input has");
  }
  if (updatesRank != indicesRank - 1 + inputRank - numSlicedDims) {
    return ShapeMaterializeResult::error(
        "[scatter_nd] requires (updates_rank == indices_rank - 1 + input_rank "
        "- num_sliced_dims)");
  }

  // Check dimensions of input shapes.
  for (int64_t i = 0; i < indicesRank - 1; i++) {
    Var indicesDim = b.createVar(indicesShape[i]);
    Var updatesDim = b.createVar(updatesShape[i]);
    b.createAssert(
        indicesDim == updatesDim,
        "[scatter_nd] batch dimensions of indices and updates don't match");
  }
  for (int64_t i = 0; i < inputRank - numSlicedDims; i++) {
    Var inputDim = b.createVar(inputShape[i + numSlicedDims]);
    Var updatesDim = b.createVar(updatesShape[i + indicesRank - 1]);
    b.createAssert(
        inputDim == updatesDim,
        "[scatter_nd] updated dimensions of input and updates don't match");
  }

  // Create and return output shape.
  return ShapeMaterializeResult::success(std::move(paramBindings), inputShape);
}

/// Materialize the shape logic for `MO::ReshapeOp`.
///
/// Specifically, if target shape contains a `-1`, infer it based on the
/// invariant that `inputShapeAttr` and `targetShapeAttr` represent equal number
/// of elements.
///
/// allowParametricDenominator removes the `denominator != 0` check when
/// parametric expressions are involved (e.g. [-1, X]). The default shape
/// inference behavour is very pessimistic. It assumes that `X` might be `0`,
/// which fails the check. This flag allows a more optimistic view where `X` is
/// assumed to not be `0` (requires runtime assert to be guaranteed).
///
/// Example:
///   inputShape = [3, D4]
///   targetShape = [D0, -1, 42]
///   returns [D0, RS0, 42]
/// with the following operation accumulated in `b`
///   kgen.param.declare RS0 = <div(mul(3, D4), mul(D0, 42)>
ShapeMaterializeResult
materializeReshapeShape(GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
                        TypedAttr targetShapeAttr,
                        bool allowParametricDenominator) {
  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // Make sure both input and target shapes are ranked symbolic shapes.
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr targetShape = dyn_cast<ShapeAttr>(targetShapeAttr);
  if (!inputShape || !targetShape)
    return ShapeMaterializeResult::nonMaterializable();

  // Find index of `-1` dimension (if any), and collect all other dimensions.
  SmallVector<TypedAttr> targetNonNegOneDims;
  std::optional<size_t> maybeTargetNegOneDimIdx = std::nullopt;
  for (auto [idx, targetDim] : llvm::enumerate(targetShape.getValues())) {
    auto intDim = dyn_cast<IntegerAttr>(targetDim);
    if (intDim && intDim.getInt() == -1) {
      // Detected multiple negative one, gives up and let runtime handle the
      // error via a shape function call.
      if (maybeTargetNegOneDimIdx.has_value()) {
        return ShapeMaterializeResult::error(
            "[reshape] multiple -1 detected in target shape");
      }

      maybeTargetNegOneDimIdx = idx;
    } else {
      // TODO: we are using the assumption that all non-integer attribute
      // (e.g., `D0` or `mul(D0, D1)`) will be non-negative. To make this
      // completely safe, we'll need to extend and leverage the range
      // analysis here.
      targetNonNegOneDims.push_back(targetDim);
    }
  }

  // resolve the `-1` in output shape if needed.
  ShapeAttr outputShape = targetShape;
  if (maybeTargetNegOneDimIdx.has_value()) {
    // Now infer the actual value of `-1` dimension based on other dimensions
    Var numerator = b.createVar(inputShape.getSymbolicNumElements());

    // Add a 1 to the shapes to multiply to ensure the denominator is valid
    auto mulDims = targetNonNegOneDims;
    mulDims.emplace_back(IntegerAttr::get(numerator.getAttr().getType(), 1));
    Var denominator =
        b.createVar(KGEN::ParamOperatorAttr::get(KGEN::POC::MulNuw, mulDims));

    if (allowParametricDenominator) {
      for (auto dim : mulDims) {
        auto intDim = dyn_cast<IntegerAttr>(dim);
        if (intDim && intDim.getInt() == 0) {
          // Guaranteed failure either way.
          return ShapeMaterializeResult::error(
              "[reshape] cannot infer dimension when a specified output "
              "dimension is 0");
        }
      }
    } else {
      // Make sure we check for 0 denominator
      b.createAssert(denominator != 0,
                     "[reshape] cannot infer dimension when a specified output "
                     "dimension is 0");
    }

    Var resolvedDim = numerator / denominator;
    SmallVector<TypedAttr> resolvedDims = targetNonNegOneDims;
    resolvedDims.insert(resolvedDims.begin() + maybeTargetNegOneDimIdx.value(),
                        resolvedDim.getAttr());
    outputShape = b.getShapeAttr(resolvedDims);
  }

  // Insert assertions
  Var inputNumeElems = b.createVar(inputShape.getSymbolicNumElements());
  Var outputNumeElems = b.createVar(outputShape.getSymbolicNumElements());
  b.createAssert(inputNumeElems == outputNumeElems,
                 "[reshape] input and output number of elements must match");
  return ShapeMaterializeResult::success(std::move(paramBindings), outputShape);
}

ShapeMaterializeResult matierlizeRoiAlignShape(GraphOp graphOp, Location loc,
                                               TypedAttr inputShapeAttr,
                                               TypedAttr roisShapeAttr,
                                               TypedAttr outHeightAttr,
                                               TypedAttr outWidthAttr) {
  // Make sure input and rois shapes are ranked.
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr roisShape = dyn_cast<ShapeAttr>(roisShapeAttr);
  if (!inputShape || !roisShape)
    return ShapeMaterializeResult::nonMaterializable();

  // Basic rank check.
  if (inputShape.getRank() != 4) {
    return ShapeMaterializeResult::error(
        "[roi_align] input shape must have rank 4");
  }
  if (roisShape.getRank() != 2) {
    return ShapeMaterializeResult::error(
        "[roi_align] rois shape must have rank 2");
  }
  if (!outHeightAttr.getType().isIntOrIndex()) {
    return ShapeMaterializeResult::error(
        "[roi_align] output height must be a single integer");
  }
  if (!outWidthAttr.getType().isIntOrIndex()) {
    return ShapeMaterializeResult::error(
        "[roi_align] output width must be a single integer");
  }

  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // Check height and width
  Var outHeight = b.createVar(outHeightAttr);
  Var outWidth = b.createVar(outWidthAttr);
  b.createAssert(outHeight >= 0,
                 "[roi_align] output height must be non-negative");
  b.createAssert(outWidth >= 0,
                 "[roi_align] output width must be non-negative");

  // Create and return output shape
  ShapeAttr outShape = b.getShapeAttr(
      {roisShape[0], outHeightAttr, outWidthAttr, inputShape[3]});
  return ShapeMaterializeResult::success(std::move(paramBindings), outShape);
}

/// Materialize the shape logic for `MO::SliceOp`.
ShapeMaterializeResult materializeSliceShape(GraphOp graphOp, Location loc,
                                             TypedAttr inputShapeAttr,
                                             TypedAttr startsAttr,
                                             TypedAttr stopsAttr,
                                             TypedAttr stepsAttr) {
  // Make sure input shape is ranked, and bounds are list of symbolic integers
  // with known sizes.
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr starts = dyn_cast<ShapeAttr>(startsAttr);
  ShapeAttr stops = dyn_cast<ShapeAttr>(stopsAttr);
  ShapeAttr steps = dyn_cast<ShapeAttr>(stepsAttr);
  if (!inputShape || !starts || !stops || !steps)
    return ShapeMaterializeResult::nonMaterializable();

  size_t inputRank = inputShape.getRank();
  if (starts.getSize() != inputRank ||
      !llvm::all_equal({starts.getSize(), stops.getSize(), steps.getSize()})) {
    return ShapeMaterializeResult::error(
        "Input rank must match the number of elements in starts/stops/steps "
        "bounds");
  }

  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // Compute and collect output spatial dimensions.
  SmallVector<TypedAttr> outDims;
  for (size_t axis = 0; axis < inputRank; axis++) {
    Var inDim = b.createVar(inputShape[axis]);
    Var start = b.createVar(starts[axis]);
    Var stop = b.createVar(stops[axis]);
    Var step = b.createVar(steps[axis]);

    // Check step != 0
    b.createAssert(step != 0, "[slice] step must be non-zero");

    // Compute the min/max for clamping start/end
    Var idxMin = b.createCond(step > 0, b.createVar(0), b.createVar(-1));
    Var idxMax = b.createCond(step > 0, inDim, inDim - 1);

    // Normalize and clamp start/stop values.
    start = b.createCond(start < 0, start + inDim, start);
    start = b.createMax(idxMin, b.createMin(idxMax, start));
    stop = b.createCond(stop < 0, stop + inDim, stop);
    stop = b.createMax(idxMin, b.createMin(idxMax, stop));

    // Make sure start and stop don't cross each other.
    b.createAssert(step < 0 || start <= stop,
                   "[slice] normalized stop cannot be smaller than start for "
                   "positive step");
    b.createAssert(step > 0 || start >= stop,
                   "[slice] normalized start cannot be smaller than stop for "
                   "negative step");

    Var outDim = computeRangeSize(start, stop, step);
    outDims.push_back(outDim.getAttr());
  }

  ShapeAttr outputShape = b.getShapeAttr(outDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), outputShape);
}

ShapeMaterializeResult materializeSplitOutputShapes(GraphOp graphOp,
                                                    Location loc,
                                                    TypedAttr inputShapeAttr,
                                                    TypedAttr splitSizesAttr,
                                                    TypedAttr axisAttr) {
  // Make sure input is ranked.
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr splitSizes = dyn_cast<ShapeAttr>(splitSizesAttr);
  if (!inputShape || !splitSizes)
    return ShapeMaterializeResult::nonMaterializable();

  // Check for constant axis
  std::optional<int64_t> axisOr = tryGetConstantInt(axisAttr);
  if (!axisOr.has_value())
    return ShapeMaterializeResult::nonMaterializable();
  int64_t axis = axisOr.value();

  // Normalize axis and check it's in range.
  size_t inputRank = inputShape.getRank();
  axis = axis < 0 ? axis + inputRank : axis;
  if (axis < 0 || inputRank <= static_cast<size_t>(axis)) {
    return ShapeMaterializeResult::error(
        "[split] normalized axis must be within range [0, rank)");
  }

  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // Check for sum of splitSizes.
  Var splitSizesSum = b.createVar(0);
  for (TypedAttr splitSize : splitSizes.getValues())
    splitSizesSum = splitSizesSum + b.createVar(splitSize);
  b.createAssert(
      splitSizesSum == b.createVar(inputShape[axis]),
      "[split] sum of split sizes must match input dimension at split axis");

  // Create output shape.
  SmallVector<ShapeAttr> outShapes;
  for (size_t i = 0; i < splitSizes.getRank(); i++) {
    SmallVector<TypedAttr> outDims(inputShape.getValues());
    outDims[axis] = splitSizes[i];
    ShapeAttr outputShape = b.getShapeAttr(outDims);
    outShapes.push_back(outputShape);
  }
  return ShapeMaterializeResult::success(std::move(paramBindings),
                                         std::move(outShapes));
}

ShapeMaterializeResult
materializeSqueezeShapeShape(GraphOp graphOp, Location loc,
                             TypedAttr inputShapeAttr,
                             TypedAttr indicesShapeAttr) {
  // Make sure both lhs and rhs shapes are ranked.
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr indicesShape = dyn_cast<ShapeAttr>(indicesShapeAttr);
  if (!inputShape || !indicesShape)
    return ShapeMaterializeResult::nonMaterializable();

  // Simple rank check.
  if (inputShape.getRank() != 1 || indicesShape.getRank() != 1) {
    return ShapeMaterializeResult::error(
        "[squeeze_shape] inputs must be rank 1 shape tensors");
  }

  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);
  Var inputNumElems = b.createVar(inputShape[0]);
  Var indicesNumElems = b.createVar(indicesShape[0]);
  b.createAssert(
      inputNumElems >= indicesNumElems,
      "[squeeze_shape] cannot remove more dimensions than there exists");

  // Create output shape.
  Var outputNumElems = inputNumElems - indicesNumElems;
  ShapeAttr outShape = b.getShapeAttr(outputNumElems);
  return ShapeMaterializeResult::success(std::move(paramBindings), outShape);
}

ShapeMaterializeResult materializeTileShape(GraphOp graphOp, Location loc,
                                            TypedAttr inputShapeAttr,
                                            TypedAttr repeatsAttr) {
  auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  auto repeats = dyn_cast<ShapeAttr>(repeatsAttr);
  // Make sure input shape is ranked and repeats is a list of symbolic integers
  // with known length.
  if (!inputShape || !repeats)
    return ShapeMaterializeResult::nonMaterializable();

  // Preliminary rank checks.
  size_t inputRank = inputShape.getRank();
  if (inputRank != repeats.getSize()) {
    return ShapeMaterializeResult::error(
        "[tile] requires (len(repeats) == input_rank)");
  }

  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // Compute and collect output dimensions.
  SmallVector<TypedAttr> resDims;
  for (size_t axis = 0; axis < inputRank; axis++) {
    Var inputDim = b.createVar(inputShape[axis]);
    Var repeat = b.createVar(repeats[axis]);
    Var outputDim = inputDim * repeat;
    resDims.push_back(outputDim.getAttr());
  }

  // Create output shape
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

ShapeMaterializeResult materializeTransposeShape(GraphOp graphOp, Location loc,
                                                 TypedAttr inputShapeAttr,
                                                 TypedAttr permutationsAttr) {
  // Make sure input shape is ranked.
  auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  if (!inputShape)
    return ShapeMaterializeResult::nonMaterializable();

  // Preliminary rank checks.
  auto permutations = dyn_cast<ShapeAttr>(permutationsAttr);
  size_t inputRank = inputShape.getRank();
  if (inputRank != permutations.getSize()) {
    return ShapeMaterializeResult::error(
        "[transpose] permutation size must match input rank");
  }

  // Make sure permutations is a list of constant integers with known length.
  SmallVector<int64_t> permInts;
  for (TypedAttr permAttr : permutations.getValues()) {
    if (auto permInt = tryGetConstantInt(permAttr)) {
      permInts.push_back(permInt.value());
    } else {
      return ShapeMaterializeResult::nonMaterializable();
    }
  }

  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // Compute and collect output dimensions.
  SmallVector<TypedAttr> resDims;
  for (size_t axis = 0; axis < inputRank; axis++) {
    // Retrieve and validate the constant permutation index.
    int64_t perm = permInts[axis];
    if (perm < 0 || inputRank <= static_cast<size_t>(perm)) {
      return ShapeMaterializeResult::error(
          "[transpose] each permutation must be within range [0, rank)");
    }
    // Retrieve the output dimension.
    TypedAttr outputDim = inputShape[perm];
    resDims.push_back(outputDim);
  }

  // Create output shape
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

ShapeMaterializeResult materializeTopKShape(GraphOp graphOp, Location loc,
                                            TypedAttr inputShape, TypedAttr k,
                                            TypedAttr axis) {
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  auto axisOr = tryGetConstantInt(axis);
  auto inputShapeAttr = dyn_cast<ShapeAttr>(inputShape);

  if (!axisOr.has_value() || !inputShapeAttr)
    return ShapeMaterializeResult::nonMaterializable();

  int64_t rank = inputShapeAttr.getRank();
  if (*axisOr < 0 || *axisOr >= rank)
    return ShapeMaterializeResult::error(
        "[top/bottom-k] axis must be within [0, rank)");

  auto kVar = b.createVar(k);
  auto dimVar = b.createVar(inputShapeAttr.getValues()[*axisOr]);
  b.createAssert(kVar >= 0 && kVar <= dimVar,
                 "[top/bottom-k] k must be between [0, dim_size].");

  SmallVector<TypedAttr> finalShape(inputShapeAttr.getValues());
  finalShape[*axisOr] = k;
  return ShapeMaterializeResult::success(std::move(paramBindings),
                                         b.getShapeAttr(finalShape));
}

ShapeMaterializeResult
materializeUnsqueezeShapeShape(GraphOp graphOp, Location loc,
                               TypedAttr inputShapeAttr,
                               TypedAttr indicesShapeAttr) {
  // Make sure both lhs and rhs shapes are ranked.
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr indicesShape = dyn_cast<ShapeAttr>(indicesShapeAttr);
  if (!inputShape || !indicesShape)
    return ShapeMaterializeResult::nonMaterializable();

  // Simple rank check.
  if (inputShape.getRank() != 1 || indicesShape.getRank() != 1) {
    return ShapeMaterializeResult::error(
        "[squeeze_shape] inputs must be rank 1 shape tensors");
  }

  // Create builder and track generated parameter bindings.
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);
  Var inputNumElems = b.createVar(inputShape[0]);
  Var indicesNumElems = b.createVar(indicesShape[0]);

  // Create output shape
  ShapeAttr outShape = b.getShapeAttr(inputNumElems + indicesNumElems);
  return ShapeMaterializeResult::success(std::move(paramBindings), outShape);
}

//------------------------Shape functions end---------------------------------//
//----------------------------------------------------------------------------//

/// Try to unpack the given attribute as a `KGEN::ParamOperatorAttr` that is
/// representing a `mo.assert`, and return the `(condAttr, errorMsgAttr)` pair.
static std::optional<std::pair<TypedAttr, StringAttr>>
tryGetAssertCondAndMsg(TypedAttr attr) {
  // Check it's an expression attribute.
  auto paramOperatorAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr);
  if (!paramOperatorAttr)
    return std::nullopt;

  // Check it's a `mo.assert`.
  auto mlirOpAttr = dyn_cast<KGEN::MLIROpAttr>(paramOperatorAttr.getOperand(0));
  if (!mlirOpAttr || mlirOpAttr.getName() != MO::AssertOp::getOperationName())
    return std::nullopt;

  // Package the condition attribute and error message.
  TypedAttr cond = paramOperatorAttr.getOperand(1);
  auto msgAttr = mlirOpAttr.getAttrs().getAs<StringAttr>("message");
  return std::make_pair(cond, msgAttr);
}

StringAttr tryGetAssertFalseErrMsg(TypedAttr attr) {
  if (auto condAndMsg = tryGetAssertCondAndMsg(attr)) {
    TypedAttr cond = condAndMsg.value().first;
    StringAttr errMsg = condAndMsg.value().second;

    // Check assert condition is a static false.
    if (auto boolAttr = dyn_cast<BoolAttr>(cond);
        boolAttr && boolAttr.getValue() == false) {
      return errMsg;
    }
  }
  return {};
}

/// Implementation of `propagateShape` which doesn't erase `bindings`.
ErrorOr<SmallVector<ShapeAttr>>
propagateShape(ArrayRef<ParamBinding> bindings,
               ArrayRef<ShapeAttr> targetShapes) {

  // Setting things up
  KGEN::ParameterEvaluator evaluator;
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](KGEN::ParamDeclRefAttr declRef) -> Attribute {
    Attribute resolvedAttr = evaluator.getReboundAttribute(declRef);
    return resolvedAttr;
  });

  // Process individual ops and propagate by simple substitution.
  for (const ParamBinding &binding : bindings) {
    KGEN::ParamDeclAttr lhsAttr = binding.decl;
    TypedAttr rhsAttr = binding.value;

    // Apply accumulated substitutions to the binding value.
    rhsAttr = cast<TypedAttr>(replacer.replace(rhsAttr));

    // Check for assert.
    if (auto condAndMsg = tryGetAssertCondAndMsg(rhsAttr)) {
      TypedAttr cond = condAndMsg.value().first;
      StringAttr errMsg = condAndMsg.value().second;

      // Error if condition didn't fold to true.
      if (auto boolAttr = dyn_cast<BoolAttr>(cond);
          !boolAttr || !boolAttr.getValue()) {
        return Error(errMsg.str());
      }
    }

    // Otherwise remember this substitution.
    evaluator.setParameterValue(lhsAttr, rhsAttr);
  }

  // Apply accumulated substitutions to target shape.
  SmallVector<ShapeAttr> resultShapes;
  for (ShapeAttr targetShape : targetShapes) {
    auto resultShape = cast<ShapeAttr>(replacer.replace(targetShape));
    resultShapes.push_back(resultShape);
  }
  return resultShapes;
}

// TODO move the following into a new interface.

// Assume `inputs` are staticized inputs for a `MO::RoiAlignOp`, try
// materializing the its shape logic.
static ShapeMaterializeResult
matierlizeRoiAlignShapeImpl(ArrayRef<TypedAttr> inputs,
                            DictionaryAttr kgenParams, GraphOp graphOp,
                            Location loc) {
  TypedAttr inputShape = inputs[0];
  TypedAttr roisShape = inputs[1];
  TypedAttr outHeight = inputs[2];
  TypedAttr outWidth = inputs[3];
  return matierlizeRoiAlignShape(graphOp, loc, inputShape, roisShape, outHeight,
                                 outWidth);
}

// Assume `inputs` are staticized inputs for a `MO::RandomNormalOp`, try
// materializing the its shape logic.
static ShapeMaterializeResult
materializeRandomShapeImpl(ArrayRef<TypedAttr> inputs,
                           DictionaryAttr kgenParams, GraphOp graphOp,
                           Location loc) {
  return materializeTargetShapeLike(graphOp, loc, inputs[0]);
}

// Assume `inputs` are staticized inputs for a `MO::RandomNormalOp`, try
// materializing the its shape logic.
static ShapeMaterializeResult
materializeResizeShapeImpl(ArrayRef<TypedAttr> inputs,
                           DictionaryAttr kgenParams, GraphOp graphOp,
                           Location loc) {
  return materializeTargetShapeLike(graphOp, loc, inputs[1]);
}

// Assume `inputs` are staticized inputs for a `MO::ScatterNd`-family op, try
// materializing the its shape logic.
static ShapeMaterializeResult
materializeScatterNdShapeShapeImpl(ArrayRef<TypedAttr> inputs,
                                   DictionaryAttr kgenParams, GraphOp graphOp,
                                   Location loc) {
  TypedAttr inputShape = inputs[0];
  TypedAttr updatesShape = inputs[1];
  TypedAttr indicesShape = inputs[2];
  return materializeScatterNdShape(graphOp, loc, inputShape, updatesShape,
                                   indicesShape);
}

// Assume `inputs` are staticized inputs for a `MO::SliceOp`, try
// materializing the its shape logic.
static ShapeMaterializeResult
materializeSliceShapeImpl(ArrayRef<TypedAttr> inputs, DictionaryAttr kgenParams,
                          GraphOp graphOp, Location loc) {
  TypedAttr input = inputs[0];
  TypedAttr starts = inputs[1];
  TypedAttr stops = inputs[2];
  TypedAttr steps = inputs[3];
  return materializeSliceShape(graphOp, loc, input, starts, stops, steps);
}

// Assume `inputs` are staticized inputs for a `MO::SqueezeShapeOp`, try
// materializing the its shape logic.
static ShapeMaterializeResult
materializeSqueezeShapeShapeImpl(ArrayRef<TypedAttr> inputs,
                                 DictionaryAttr kgenParams, GraphOp graphOp,
                                 Location loc) {
  TypedAttr inputShape = inputs[0];
  TypedAttr indicesShape = inputs[1];
  return materializeSqueezeShapeShape(graphOp, loc, inputShape, indicesShape);
}

// Assume `inputs` are staticized inputs for a `MO::TileOp`, try materializing
// the its shape logic.
static ShapeMaterializeResult
materializeTileShapeImpl(ArrayRef<TypedAttr> inputs, DictionaryAttr kgenParams,
                         GraphOp graphOp, Location loc) {
  TypedAttr inputShape = inputs[0];
  TypedAttr repeats = inputs[1];
  return materializeTileShape(graphOp, loc, inputShape, repeats);
}

// Assume `inputs` are staticized inputs for a `MO::TopKOp` or `MO::BottomKOp`,
// try materializing the its shape logic.
static ShapeMaterializeResult
materializeTopKShapeImpl(ArrayRef<TypedAttr> inputs, DictionaryAttr kgenParams,
                         GraphOp graphOp, Location loc) {
  TypedAttr inputShape = inputs[0];
  TypedAttr k = inputs[1];
  TypedAttr axis = inputs[2];
  return materializeTopKShape(graphOp, loc, inputShape, k, axis);
}

// Assume `inputs` are staticized inputs for a `MO::TransposeOp`, try
// materializing the its shape logic.
static ShapeMaterializeResult
materializeTransposeShapeImpl(ArrayRef<TypedAttr> inputs,
                              DictionaryAttr kgenParams, GraphOp graphOp,
                              Location loc) {
  TypedAttr inputShape = inputs[0];
  TypedAttr permutations = inputs[1];
  return materializeTransposeShape(graphOp, loc, inputShape, permutations);
}

// Assume `inputs` are staticized inputs for a `MO::UnsqueezeShapeOp`, try
// materializing the its shape logic.
static ShapeMaterializeResult
materializeUnsqueezeShapeShapeImpl(ArrayRef<TypedAttr> inputs,
                                   DictionaryAttr kgenParams, GraphOp graphOp,
                                   Location loc) {
  TypedAttr inputShape = inputs[0];
  TypedAttr indicesShape = inputs[1];
  return materializeUnsqueezeShapeShape(graphOp, loc, inputShape, indicesShape);
}

std::optional<MaterializeShapeFuncTy>
tryGetShapeMaterializeFunc(StringRef shapeFuncName) {
  // A hard-coded map from shape function name to how their shape logic can be
  // materialized.
  static DenseMap<StringRef, MaterializeShapeFuncTy> nameToFunc = {
      {"random_shape", materializeRandomShapeImpl},
      {"resize_shape", materializeResizeShapeImpl},
      {"roi_align_shape", matierlizeRoiAlignShapeImpl},
      {"scatter_nd_shape", materializeScatterNdShapeShapeImpl},
      {"slice_shape", materializeSliceShapeImpl},
      {"squeeze_shape_shape", materializeSqueezeShapeShapeImpl},
      {"tile_shape", materializeTileShapeImpl},
      {"top_k_shape", materializeTopKShapeImpl},
      {"transpose_shape", materializeTransposeShapeImpl},
      {"unsqueeze_shape_shape", materializeUnsqueezeShapeShapeImpl},
  };

  auto iter = nameToFunc.find(shapeFuncName);
  if (iter == nameToFunc.end())
    return std::nullopt;
  return iter->getSecond();
}

} // namespace M::MO
