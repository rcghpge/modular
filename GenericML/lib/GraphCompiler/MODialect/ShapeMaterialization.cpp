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
// Linalg::SolveOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> LinAlg::SolveOp::getDataDependentInputIndices() {
  return {};
}

ShapeMaterializeResult
LinAlg::SolveOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  auto lhsShape = inputAttrs[0];
  auto rhsShape = inputAttrs[1];

  return materializeMatrixSolveShape(graphOp, loc, lhsShape, rhsShape);
}

//===----------------------------------------------------------------------===//
// Pad*Op
//===----------------------------------------------------------------------===//

SmallVector<int32_t> PadConstantOp::getDataDependentInputIndices() {
  return {1};
}
SmallVector<int32_t> PadReflectOp::getDataDependentInputIndices() {
  return {1};
}
SmallVector<int32_t> PadRepeatOp::getDataDependentInputIndices() { return {1}; }

static ShapeMaterializeResult
commonPadMaterializeShapeDefs(Operation *op, ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = op->getParentOfType<GraphOp>();
  Location loc = op->getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr paddings = inputAttrs[1];

  return materializePadShape(graphOp, loc, inputShape, paddings);
}

ShapeMaterializeResult
PadConstantOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return commonPadMaterializeShapeDefs(getOperation(), inputAttrs);
}
ShapeMaterializeResult
PadReflectOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return commonPadMaterializeShapeDefs(getOperation(), inputAttrs);
}
ShapeMaterializeResult
PadRepeatOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return commonPadMaterializeShapeDefs(getOperation(), inputAttrs);
}

//===----------------------------------------------------------------------===//
// *PoolOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> MaxPoolOp::getDataDependentInputIndices() {
  return {1, 2, 3, 4};
}
SmallVector<int32_t> AvgPoolOp::getDataDependentInputIndices() {
  return {1, 2, 3, 4};
}
SmallVector<int32_t> MaxPoolCeilOp::getDataDependentInputIndices() {
  return {1, 2, 3, 4};
}
SmallVector<int32_t> AvgPoolCeilOp::getDataDependentInputIndices() {
  return {1, 2, 3, 4};
}

static ShapeMaterializeResult
commonPoolMaterializeShapeDefs(Operation *op, ArrayRef<TypedAttr> inputAttrs,
                               bool ceilMode) {
  auto graphOp = op->getParentOfType<GraphOp>();
  Location loc = op->getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr filter = inputAttrs[1];
  TypedAttr strides = inputAttrs[2];
  TypedAttr dilations = inputAttrs[3];
  TypedAttr paddings = inputAttrs[4];

  return materializePoolShape(graphOp, loc, inputShape, filter, strides,
                              dilations, paddings, ceilMode);
}

ShapeMaterializeResult
MaxPoolOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return commonPoolMaterializeShapeDefs(getOperation(), inputAttrs, false);
}
ShapeMaterializeResult
AvgPoolOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return commonPoolMaterializeShapeDefs(getOperation(), inputAttrs, false);
}
ShapeMaterializeResult
MaxPoolCeilOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return commonPoolMaterializeShapeDefs(getOperation(), inputAttrs, true);
}
ShapeMaterializeResult
AvgPoolCeilOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return commonPoolMaterializeShapeDefs(getOperation(), inputAttrs, true);
}

//===----------------------------------------------------------------------===//
// RandomOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> RandomNormalOp::getDataDependentInputIndices() {
  return {0};
}

ShapeMaterializeResult
RandomNormalOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  return materializeTargetShapeLike(graphOp, loc, inputAttrs[0]);
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
// Resize*Op
//===----------------------------------------------------------------------===//

SmallVector<int32_t> ResizeLinearOp::getDataDependentInputIndices() {
  return {1};
}
SmallVector<int32_t> ResizeNearestOp::getDataDependentInputIndices() {
  return {1};
}

ShapeMaterializeResult
ResizeLinearOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  return materializeTargetShapeLike(graphOp, loc, inputAttrs[1]);
}
ShapeMaterializeResult
ResizeNearestOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  return materializeTargetShapeLike(graphOp, loc, inputAttrs[1]);
}

//===----------------------------------------------------------------------===//
// ROIAlignOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> RoiAlignOp::getDataDependentInputIndices() {
  return {2, 3};
}

ShapeMaterializeResult
RoiAlignOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr roisShape = inputAttrs[1];
  TypedAttr outHeight = inputAttrs[2];
  TypedAttr outWidth = inputAttrs[3];
  return materializeRoiAlignShape(graphOp, loc, inputShape, roisShape,
                                  outHeight, outWidth);
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
