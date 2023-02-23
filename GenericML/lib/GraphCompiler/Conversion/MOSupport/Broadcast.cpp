//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOSupport/Broadcast.h"

namespace M::MO {

MOTensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                    MOTensorValue from, TensorType reshapeType,
                                    MOTensorValue to) {
  Type elemType = builder.getIntegerType(64, /*isSigned=*/true);

  assert(reshapeType.hasRank() &&
         "reshapeType needs to be a statically ranked tensor type");
  auto type = MO::TensorType::get({(int64_t)reshapeType.getRank()}, elemType);

  auto valueAttr = builder.getAttr<IntArrayElementsAttr>(
      ArrayRef<int64_t>(reshapeType.getIntDims()), mlir::IntegerType::Signed);

  auto reshapeShapeValue = builder.create<MO::ConstantOp>(loc, type, valueAttr);

  auto reshapedValue =
      builder.create<MO::ReshapeOp>(loc, reshapeType, from, reshapeShapeValue);

  return broadcastTo(builder, loc, reshapedValue, to);
}

MOTensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                    MOTensorValue from,
                                    ArrayRef<int64_t> reshapeShape,
                                    MOTensorValue to) {
  TensorType type = from.getType();
  TensorType reshapeType =
      MO::TensorType::get(reshapeShape, type.getElementType());
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

/// Computes the common broadcast shape of `from` and `to`.
/// Supports parametric and unknown shapes, but not dynamic rank.
static ErrorOr<MO::TensorType> broadcastShape(MO::TensorType from,
                                              MO::TensorType to) {
  // This largely should follow the semantics described here:
  // https://numpy.org/doc/stable/user/basics.broadcasting.html
  // TODO (#7331): Merge with Support/ML/BroadcastShape.

  if (!from.hasRank() || !to.hasRank())
    return Error(Twine("Guards should have rejected unranked inputs."));

  auto idxTy = IndexType::get(from.getContext());
  auto intDim1 = IntegerAttr::get(idxTy, 1ll);

  SmallVector<TypedAttr> result;
  SmallVector<TypedAttr> fromDims(from.getDims());
  SmallVector<TypedAttr> toDims(to.getDims());
  for (auto [fromDim, toDim] :
       llvm::zip_longest(llvm::reverse(fromDims), llvm::reverse(toDims))) {
    if (!toDim.has_value()) {
      result.push_back(fromDim.value());
      continue;
    }
    if (!fromDim.has_value()) {
      result.push_back(toDim.value());
      continue;
    }

    TypedAttr fromDimVal = fromDim.value();
    TypedAttr toDimVal = toDim.value();

    if (fromDimVal == toDimVal) {
      result.push_back(fromDimVal);
      continue;
    }

    if (fromDimVal == intDim1) {
      result.push_back(toDimVal);
      continue;
    }
    if (toDimVal == intDim1) {
      result.push_back(fromDimVal);
      continue;
    }

    if (fromDim->isa<KGEN::ParamDeclRefAttr>()) {
      if (toDim->isa<KGEN::UnknownAttr>()) {
        return Error(Twine("fromDim and toDim are incompatible"));
      }
    }

    if (toDim->isa<KGEN::ParamDeclRefAttr>()) {
      if (fromDim->isa<KGEN::UnknownAttr>()) {
        return Error(Twine("fromDim and toDim are incompatible"));
      }
    }

    if (fromDim->isa<KGEN::UnknownAttr>()) {
      result.push_back(toDimVal);
      continue;
    }

    if (toDim->isa<KGEN::UnknownAttr>()) {
      result.push_back(fromDimVal);
      continue;
    }

    // Not 1, not dynamic. Then the other shape necessarily matches this.
    if (fromDimVal != toDimVal) {
      return Error(Twine(
          "upstream dialect should have rejected non-broadcastable inputs"));
    }
    result.push_back(fromDimVal);
  }

  return MO::TensorType::get(SmallVector<TypedAttr>(llvm::reverse(result)),
                             from.getElementType());
}

/// Broadcast `x` and `y` to a common dynamic shape given by `bcastCommonShape`.
static std::pair<MOTensorValue, MOTensorValue>
broadcastToCommonShape(OpBuilder &builder, MOTensorValue x, MOTensorValue y,
                       MO::TensorType bcastCommonShape) {
  auto intType = builder.getIntegerType(64, /*isSigned=*/true);
  auto shape =
      MO::TensorType::get({(int64_t)bcastCommonShape.getRank()}, intType);

  // Get the shapes of the inputs at runtime
  TensorType xType = x.getType();
  assert(xType.hasRank());

  auto xShape = builder.create<MO::ShapeOfOp>(
      x.getLoc(), MO::TensorType::get({(int64_t)xType.getRank()}, intType), x);

  TensorType yType = y.getType();
  assert(yType.hasRank());
  auto yShape = builder.create<MO::ShapeOfOp>(
      y.getLoc(), MO::TensorType::get({(int64_t)yType.getRank()}, intType), y);

  auto bcastShape = builder.create<MO::BroadcastShapeOp>(
      x.getLoc(), shape, xShape.getShape(), yShape.getShape());

  auto bcastXOp = builder.create<MO::BroadcastToOp>(
      x.getLoc(), bcastCommonShape, x, bcastShape);
  auto bcastYOp = builder.create<MO::BroadcastToOp>(
      x.getLoc(), bcastCommonShape, y, bcastShape);
  return {bcastXOp, bcastYOp};
}

/// Broadcast `x` to a static shape given by `s`.
static MOTensorValue broadcastToStaticShape(OpBuilder &builder, Location loc,
                                            MOTensorValue x, MO::TensorType s) {
  auto constOp = builder.create<MO::ConstantOp>(
      loc, s.getIntDims(), builder.getIntegerType(64, /*isSigned=*/true));
  return builder.create<MO::BroadcastToOp>(loc, s, x, constOp);
}

std::pair<MOTensorValue, MOTensorValue>
reifyImplicitBroadcasting(OpBuilder &builder, Location loc, MOTensorValue x,
                          MOTensorValue y) {
  TensorType xShape = x.getType();
  TensorType yShape = y.getType();
  if ((xShape == yShape) && (xShape.hasStaticShape()))
    return {x, y};

  ErrorOr<MO::TensorType> errorOrBcastCommonShape =
      broadcastShape(xShape, yShape);

  assert(!errorOrBcastCommonShape.isError() &&
         "cannot compute static broadcast shape");

  MO::TensorType bcastCommonShape = *errorOrBcastCommonShape;

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

MOTensorValue broadcastTo(OpBuilder &builder, Location loc, MOTensorValue x,
                          MOTensorValue y) {
  auto [newX, newY] = reifyImplicitBroadcasting(builder, loc, x, y);

  assert((newY.getType() == unifyShapeType(y.getType())) &&
         "don't use this helper unless you can guarantee x broadcasts to y");

  return newX;
}

} // namespace M::MO
