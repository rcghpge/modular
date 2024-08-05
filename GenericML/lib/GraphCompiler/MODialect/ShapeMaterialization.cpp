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
#include "Support/AssertStream.h"

namespace M::MO {

//===----------------------------------------------------------------------===//
// InvokeShapeFuncOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> InvokeShapeFuncOp::getDataDependentInputIndices() {
  SmallVector<int32_t> idxs;
  if (ArrayAttr arrayAttr = getDataDeptTensorsAttr()) {
    for (llvm::APInt index : arrayAttr.getAsValueRange<IntegerAttr>())
      idxs.push_back(index.getLimitedValue());
  }
  return idxs;
}

ShapeMaterializeResult
InvokeShapeFuncOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  StringRef shapeFuncName = getShapeFuncName();
  auto shapeMaterializeFunc = tryGetShapeMaterializeFunc(shapeFuncName);

  // Give up if this shape function doesn't have a C++ shape function impl.
  if (!shapeMaterializeFunc.has_value())
    return ShapeMaterializeResult::nonMaterializable();

  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  DictionaryAttr kgenParams = getKgenParamsAttr();
  return shapeMaterializeFunc.value()(inputAttrs, kgenParams, graphOp, loc);
}

//===----------------------------------------------------------------------===//
// BatchMatmulOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> BatchMatmulOp::getDataDependentInputIndices() {
  return {};
}

ShapeMaterializeResult
BatchMatmulOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputA = inputAttrs[0];
  TypedAttr inputB = inputAttrs[1];

  if (inputAttrs.size() > 2) {
    auto transposeBAsBool =
        llvm::dyn_cast_or_null<mlir::BoolAttr>(inputAttrs[2]);
    ASSERT_STREAM(!transposeBAsBool || !transposeBAsBool.getValue(),
                  "Do not support staticization of transposeB.");
  }

  return materializeBatchMatmulShape(graphOp, loc, inputA, inputB);
}

//===----------------------------------------------------------------------===//
// BroadcastShapeOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> BroadcastShapeOp::getDataDependentInputIndices() {
  return {};
}

ShapeMaterializeResult
BroadcastShapeOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr lhsShape = inputAttrs[0];
  TypedAttr rhsShape = inputAttrs[1];
  return materializeBroadcastShapeShape(graphOp, loc, lhsShape, rhsShape);
}

//===----------------------------------------------------------------------===//
// BroadcastToOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> BroadcastToOp::getDataDependentInputIndices() {
  return {1};
}

ShapeMaterializeResult
BroadcastToOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr targetShape = inputAttrs[1];
  return materializeBroadcastToShape(graphOp, loc, inputShape, targetShape);
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> ConcatOp::getDataDependentInputIndices() { return {0}; }

ShapeMaterializeResult
ConcatOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr axis = inputAttrs[0];
  ArrayRef<TypedAttr> inputShapes = inputAttrs.drop_front();

  return materializeConcatShape(graphOp, loc, axis, inputShapes);
}

//===----------------------------------------------------------------------===//
// ConvOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> ConvOp::getDataDependentInputIndices() {
  return {2, 3, 4, 5};
}

ShapeMaterializeResult
ConvOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr filterShape = inputAttrs[1];
  TypedAttr strides = inputAttrs[2];
  TypedAttr dilations = inputAttrs[3];
  TypedAttr paddings = inputAttrs[4];
  TypedAttr numGroup = inputAttrs[5];
  return materializeConvShape(graphOp, loc, inputShape, filterShape, strides,
                              dilations, paddings, numGroup);
}

//===----------------------------------------------------------------------===//
// ConvTransposeOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> ConvTransposeOp::getDataDependentInputIndices() {
  return {2, 3, 4, 5};
}

ShapeMaterializeResult
ConvTransposeOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr filterShape = inputAttrs[1];
  TypedAttr strides = inputAttrs[2];
  TypedAttr dilations = inputAttrs[3];
  TypedAttr paddings = inputAttrs[4];
  TypedAttr outPaddings = inputAttrs[5];
  return materializeConvTransposeShape(graphOp, loc, inputShape, filterShape,
                                       strides, dilations, paddings,
                                       outPaddings);
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> GatherOp::getDataDependentInputIndices() { return {2}; }

ShapeMaterializeResult
GatherOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr indicesShape = inputAttrs[1];
  TypedAttr axis = inputAttrs[2];
  return materializeGatherShape(graphOp, loc, inputShape, indicesShape, axis);
}

//===----------------------------------------------------------------------===//
// GatherNdOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> GatherNdOp::getDataDependentInputIndices() { return {}; }

ShapeMaterializeResult
GatherNdOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr indicesShape = inputAttrs[1];

  auto batchDims = getBatchDimsAttr().getInt();
  return materializeGatherNdShape(graphOp, loc, inputShape, indicesShape,
                                  batchDims);
}

//===----------------------------------------------------------------------===//
// RangeOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> RangeOp::getDataDependentInputIndices() {
  return {0, 1, 2};
}

ShapeMaterializeResult
RangeOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr startAttr = inputAttrs[0];
  TypedAttr stopAttr = inputAttrs[1];
  TypedAttr stepAttr = inputAttrs[2];
  return materializeRangeShape(graphOp, loc, startAttr, stopAttr, stepAttr);
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> ReshapeOp::getDataDependentInputIndices() { return {1}; }

ShapeMaterializeResult
ReshapeOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr targetShape = inputAttrs[1];
  return materializeReshapeShape(graphOp, loc, inputShape, targetShape);
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> SplitOp::getDataDependentInputIndices() { return {1, 2}; }

ShapeMaterializeResult
SplitOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr splitSizes = inputAttrs[1];
  TypedAttr axis = inputAttrs[2];
  return materializeSplitOutputShapes(graphOp, loc, inputShape, splitSizes,
                                      axis);
}

} // namespace M::MO
