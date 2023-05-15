//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOSupport/Broadcast.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"

namespace M::MO {

TensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                  TensorValue from, TensorType reshapeType,
                                  TensorValue to) {
  Type elemType = builder.getIntegerType(64, /*isSigned=*/true);

  assert(reshapeType.hasRank() &&
         "reshapeType needs to be a statically ranked tensor type");
  auto type = TensorType::get({(int64_t)reshapeType.getRank()}, elemType);

  auto valueAttr = builder.getAttr<IntArrayElementsAttr>(
      ArrayRef<int64_t>(reshapeType.getIntDims()), mlir::IntegerType::Signed);

  auto reshapeShapeValue =
      builder.create<MO::ConstantOp>(loc, type, cast<ElementsAttr>(valueAttr));

  auto reshapedValue =
      builder.create<MO::ReshapeOp>(loc, reshapeType, from, reshapeShapeValue);

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
static std::pair<TensorValue, TensorValue>
broadcastToCommonShape(OpBuilder &builder, TensorValue x, TensorValue y,
                       TensorType bcastCommonShape) {
  auto intType = builder.getIntegerType(64, /*isSigned=*/true);
  auto shape = TensorType::get({(int64_t)bcastCommonShape.getRank()}, intType);

  // Get the shapes of the inputs at runtime
  TensorType xType = x.getType();
  assert(xType.hasRank());

  auto xShape = builder.create<MO::ShapeOfOp>(
      x.getLoc(), TensorType::get({(int64_t)xType.getRank()}, intType), x);

  TensorType yType = y.getType();
  assert(yType.hasRank());
  auto yShape = builder.create<MO::ShapeOfOp>(
      y.getLoc(), TensorType::get({(int64_t)yType.getRank()}, intType), y);

  auto bcastShape = builder.create<MO::BroadcastShapeOp>(
      x.getLoc(), shape, xShape.getShape(), yShape.getShape());

  auto bcastXOp = builder.create<MO::BroadcastToOp>(
      x.getLoc(), bcastCommonShape, x, bcastShape);
  auto bcastYOp = builder.create<MO::BroadcastToOp>(
      x.getLoc(), bcastCommonShape, y, bcastShape);
  return {bcastXOp, bcastYOp};
}

TensorValue broadcastToStaticShape(OpBuilder &builder, Location loc,
                                   TensorValue x, TensorType s) {
  auto constOp = builder.create<MO::ConstantOp>(loc, s.getIntDims());
  return builder.create<MO::BroadcastToOp>(loc, s, x, constOp);
}

std::pair<TensorValue, TensorValue>
reifyImplicitBroadcasting(OpBuilder &builder, Location loc, TensorValue x,
                          TensorValue y) {
  TensorType xShape = x.getType();
  TensorType yShape = y.getType();
  if ((xShape == yShape) && (xShape.hasStaticShape()))
    return {x, y};

  ErrorOr<TensorType> errorOrBcastCommonShape =
      TensorType::broadcastShapes(xShape, yShape);

  assert(!errorOrBcastCommonShape.isError() &&
         "cannot compute static broadcast shape");

  TensorType bcastCommonShape = *errorOrBcastCommonShape;

  assert(bcastCommonShape.hasRank() &&
         "guards should have rejected dynamic ranks");
  if (!bcastCommonShape.hasStaticShape())
    return broadcastToCommonShape(builder, x, y, bcastCommonShape);

  // Static broadcast, can potentially elide some broadcasts.
  if (xShape != bcastCommonShape)
    x = broadcastToStaticShape(builder, loc, x, bcastCommonShape);
  if (yShape != bcastCommonShape)
    y = broadcastToStaticShape(builder, loc, y, bcastCommonShape);
  return {x, y};
}

TensorValue broadcastTo(OpBuilder &builder, Location loc, TensorValue x,
                        TensorValue y) {
  auto [newX, newY] = reifyImplicitBroadcasting(builder, loc, x, y);

  assert((newY.getType() == unifyShapeType(y.getType())) &&
         "don't use this helper unless you can guarantee x broadcasts to y");

  return newX;
}

} // namespace M::MO
