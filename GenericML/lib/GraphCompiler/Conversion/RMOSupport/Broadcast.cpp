//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/RMOSupport/Broadcast.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GenericML/GraphCompiler/RMODialect/RMOOps.h"
#include "Support/AssertStream.h"

using M::MO::ShapeAttr;
using M::MO::ShapeType;
using M::MO::TensorType;
using M::MO::TensorValue;

namespace M::RMO {

TensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                  TensorValue from, TensorType reshapeType,
                                  TensorValue to) {
  Type elemType = builder.getIntegerType(64, /*isSigned=*/true);

  ASSERT_STREAM(reshapeType.hasRank())
      << "reshapeType needs to be a statically ranked tensor type but it is "
      << reshapeType;
  auto type = TensorType::get({(int64_t)reshapeType.getRank()}, elemType);

  auto valueAttr = builder.getAttr<IntArrayElementsAttr>(
      ArrayRef<int64_t>(reshapeType.getIntDims()), mlir::IntegerType::Signed);

  auto reshapeShapeValue =
      builder.create<MO::ConstantOp>(loc, type, cast<ElementsAttr>(valueAttr));

  auto reshapedValue = builder.create<RMO::MOReshapeOp>(loc, reshapeType, from,
                                                        reshapeShapeValue);

  return broadcastTo(builder, loc, reshapedValue, to);
}

TensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                  TensorValue from,
                                  ArrayRef<int64_t> reshapeShape,
                                  TensorValue to) {
  TensorType type = from.getType();
  TensorType reshapeType = TensorType::get(reshapeShape, type.getElementType());
  return reshapeAndBroadcastTo(builder, loc, from, reshapeType, to);
}

TensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                  TensorValue from, TensorType reshapeType,
                                  TensorValue reshapeShape, TensorValue to) {
  auto reshapedValue =
      builder.create<RMO::MOReshapeOp>(loc, reshapeType, from, reshapeShape);
  return broadcastTo(builder, loc, reshapedValue, to);
}

TensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                  TensorValue from,
                                  ArrayRef<int64_t> reshapeDims,
                                  TensorValue reshapeShape, TensorValue to) {
  auto reshapedValue = builder.create<RMO::MOReshapeOp>(
      loc, MO::TensorType::get(reshapeDims, from.getType().getElementType()),
      from, reshapeShape);
  return broadcastTo(builder, loc, reshapedValue, to);
}

/// Function that ensures tensor types before and after broadcast are
/// comparable, by ensuring the `TensorType` is statically shaped.
static TensorType unifyShapeType(TensorType type) {
  if (!type.hasRank())
    return type;
  SmallVector<TypedAttr> result{type.getDims()};
  ShapeType shapeTy = type.getTypedShapeAttr().getType();
  auto shapedAttr = ShapeAttr::get(type.getContext(), result, shapeTy);
  return TensorType::get(shapedAttr, type.getDtype());
}

/// Broadcast `x` and `y` to a common dynamic shape given by `bcastCommonShape`.
/// All inputs must have a static rank.
static std::pair<TensorValue, TensorValue>
broadcastToDynamicShape(OpBuilder &builder, TensorValue x, TensorValue y,
                        ShapeAttr bcastCommonShape) {
  auto intType = builder.getIntegerType(64, /*isSigned=*/true);
  auto shape = TensorType::get({(int64_t)bcastCommonShape.getRank()}, intType);

  // Get the shapes of the inputs at runtime
  TensorType xType = x.getType();
  ASSERT_STREAM(xType.hasRank());

  auto xShape = builder.create<RMO::MOShapeOfOp>(
      x.getLoc(), TensorType::get({(int64_t)xType.getRank()}, intType), x);

  TensorType yType = y.getType();
  ASSERT_STREAM(yType.hasRank());
  auto yShape = builder.create<RMO::MOShapeOfOp>(
      y.getLoc(), TensorType::get({(int64_t)yType.getRank()}, intType), y);

  auto bcastShape = builder.create<RMO::MOBroadcastShapeOp>(
      x.getLoc(), shape, xShape.getShape(), yShape.getShape());

  auto bcastXOp = builder.create<RMO::MOBroadcastToOp>(
      x.getLoc(), x.getType().cloneWith(bcastCommonShape), x, bcastShape);
  auto bcastYOp = builder.create<RMO::MOBroadcastToOp>(
      y.getLoc(), y.getType().cloneWith(bcastCommonShape), y, bcastShape);
  return {bcastXOp, bcastYOp};
}

/// Broadcast `x`, `y` and `z` to a common dynamic shape given by
/// `bcastCommonShape`.
/// All inputs must have a static rank.
static std::tuple<TensorValue, TensorValue, TensorValue>
broadcastToDynamicShape(OpBuilder &builder, TensorValue x, TensorValue y,
                        TensorValue z, ShapeAttr xyCommonShape,
                        ShapeAttr bcastCommonShape) {
  auto intType = builder.getIntegerType(64, /*isSigned=*/true);
  auto xyShape = TensorType::get({(int64_t)xyCommonShape.getRank()}, intType);
  auto shape = TensorType::get({(int64_t)bcastCommonShape.getRank()}, intType);

  // Get the shapes of the inputs at runtime
  TensorType xType = x.getType();
  ASSERT_STREAM(xType.hasRank());
  auto xShape = builder.create<RMO::MOShapeOfOp>(
      x.getLoc(), TensorType::get({(int64_t)xType.getRank()}, intType), x);

  TensorType yType = y.getType();
  ASSERT_STREAM(yType.hasRank());
  auto yShape = builder.create<RMO::MOShapeOfOp>(
      y.getLoc(), TensorType::get({(int64_t)yType.getRank()}, intType), y);

  TensorType zType = z.getType();
  ASSERT_STREAM(zType.hasRank());
  auto zShape = builder.create<RMO::MOShapeOfOp>(
      z.getLoc(), TensorType::get({(int64_t)zType.getRank()}, intType), z);

  auto bcastXYShape = builder.create<RMO::MOBroadcastShapeOp>(
      x.getLoc(), xyShape, xShape.getShape(), yShape.getShape());
  auto bcastShape = builder.create<RMO::MOBroadcastShapeOp>(
      x.getLoc(), shape, bcastXYShape.getShape(), zShape.getShape());

  auto bcastXOp = builder.create<RMO::MOBroadcastToOp>(
      x.getLoc(), x.getType().cloneWith(bcastCommonShape), x, bcastShape);
  auto bcastYOp = builder.create<RMO::MOBroadcastToOp>(
      y.getLoc(), y.getType().cloneWith(bcastCommonShape), y, bcastShape);
  auto bcastZOp = builder.create<RMO::MOBroadcastToOp>(
      z.getLoc(), z.getType().cloneWith(bcastCommonShape), z, bcastShape);
  return {bcastXOp, bcastYOp, bcastZOp};
}

TensorValue broadcastToStaticShape(OpBuilder &builder, Location loc,
                                   TensorValue x, ShapeAttr s) {
  auto constOp = builder.create<MO::ConstantOp>(loc, s.getIntDims());
  return builder.create<RMO::MOBroadcastToOp>(loc, x.getType().cloneWith(s), x,
                                              constOp);
}

std::pair<TensorValue, TensorValue>
reifyImplicitBroadcasting(OpBuilder &builder, Location loc, TensorValue x,
                          TensorValue y) {
  TensorType xShape = x.getType();
  TensorType yShape = y.getType();
  if ((xShape == yShape) && (xShape.hasStaticShape()))
    return {x, y};

  ASSERT_STREAM(xShape.hasRank() && yShape.hasRank())
      << "broadcasting does not support dynamic rank";

  ErrorOr<ShapeAttr> errorOrBcastCommonShape = ShapeAttr::broadcastShapes(
      xShape.getTypedShapeAttr(), yShape.getTypedShapeAttr());

  ASSERT_STREAM(!errorOrBcastCommonShape.isError())
      << "the shapes are incompatible: " << errorOrBcastCommonShape.takeError();

  ShapeAttr bcastCommonShape = *errorOrBcastCommonShape;

  if (!bcastCommonShape.isStatic())
    return broadcastToDynamicShape(builder, x, y, bcastCommonShape);

  // Static broadcast, can potentially elide some broadcasts.
  if (xShape.getTypedShapeAttr() != bcastCommonShape)
    x = broadcastToStaticShape(builder, loc, x, bcastCommonShape);
  if (yShape.getTypedShapeAttr() != bcastCommonShape)
    y = broadcastToStaticShape(builder, loc, y, bcastCommonShape);
  return {x, y};
}

std::tuple<TensorValue, TensorValue, TensorValue>
reifyImplicitBroadcasting(OpBuilder &builder, Location loc, TensorValue x,
                          TensorValue y, TensorValue z) {
  TensorType xShape = x.getType();
  TensorType yShape = y.getType();
  TensorType zShape = z.getType();
  if ((xShape == yShape) && (yShape == zShape) && (xShape.hasStaticShape()))
    return {x, y, z};

  ASSERT_STREAM(xShape.hasRank() && yShape.hasRank() && zShape.hasRank())
      << "broadcasting does not support dynamic rank";

  ErrorOr<ShapeAttr> errorOrXYBcastCommonShape = ShapeAttr::broadcastShapes(
      xShape.getTypedShapeAttr(), yShape.getTypedShapeAttr());
  ASSERT_STREAM(!errorOrXYBcastCommonShape.isError())
      << "the shapes are incompatible: "
      << errorOrXYBcastCommonShape.takeError();
  ShapeAttr xyCommonShape = *errorOrXYBcastCommonShape;

  ErrorOr<ShapeAttr> errorOrBcastCommonShape = ShapeAttr::broadcastShapes(
      *errorOrXYBcastCommonShape, zShape.getTypedShapeAttr());

  ASSERT_STREAM(!errorOrBcastCommonShape.isError())
      << "the shapes are incompatible: " << errorOrBcastCommonShape.takeError();

  ShapeAttr bcastCommonShape = *errorOrBcastCommonShape;

  if (!bcastCommonShape.isStatic())
    return broadcastToDynamicShape(builder, x, y, z, xyCommonShape,
                                   bcastCommonShape);

  // Static broadcast, can potentially elide some broadcasts.
  if (xShape.getTypedShapeAttr() != bcastCommonShape)
    x = broadcastToStaticShape(builder, loc, x, bcastCommonShape);
  if (yShape.getTypedShapeAttr() != bcastCommonShape)
    y = broadcastToStaticShape(builder, loc, y, bcastCommonShape);
  if (zShape.getTypedShapeAttr() != bcastCommonShape)
    z = broadcastToStaticShape(builder, loc, z, bcastCommonShape);
  return {x, y, z};
}

TensorValue broadcastTo(OpBuilder &builder, Location loc, TensorValue x,
                        TensorValue y) {
  auto [newX, newY] = reifyImplicitBroadcasting(builder, loc, x, y);

  ASSERT_STREAM(newY.getType() == unifyShapeType(y.getType()))
      << "don't use this helper unless you can guarantee x broadcasts to y";

  return newX;
}

TensorValue broadcastFloatTo(OpBuilder &builder, Location loc,
                             const APFloat &scalar, TensorValue toVal) {
  auto toTy = cast<TensorType>(toVal.getType());

  auto graphOp = cast<MO::GraphOp>(toVal.getParentBlock()->getParentOp());
  auto toShape = builder.create<RMO::MOShapeOfOp>(loc, toVal, graphOp);

  auto scalarConst =
      builder.create<MO::ConstantOp>(loc, scalar, toTy.getElementType());
  return builder.create<RMO::MOBroadcastToOp>(loc, toTy, scalarConst, toShape);
}

} // namespace M::RMO
