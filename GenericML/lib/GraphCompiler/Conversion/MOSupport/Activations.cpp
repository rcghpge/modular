//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOSupport/Activations.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"

namespace M::MO {

TensorValue convertPRelu(OpBuilder &builder, Location loc, TensorValue x,
                         TensorValue alpha) {
  auto zero = builder.create<ConstantOp>(loc, 0.f);

  TensorValue xValue;
  bool cast = false;
  if (x.getType().getDtype() != zero.getType().getDtype()) {
    cast = true;
    xValue = builder.create<CastOp>(loc, x, zero.getType().getDtype());
  } else {
    xValue = x;
  }

  auto broadcastZero = builder.create<BroadcastToOp>(loc, zero, xValue);
  auto broadcastAlpha = builder.create<BroadcastToOp>(loc, alpha, xValue);
  auto min = builder.create<MinOp>(loc, xValue, broadcastZero);
  auto minXAlpha = builder.create<MulOp>(loc, broadcastAlpha, min);
  auto max = builder.create<MaxOp>(loc, xValue, broadcastZero);

  auto result = builder.create<AddOp>(loc, max, minXAlpha);
  if (cast) {
    return builder.create<CastOp>(loc, result, x.getType().getDtype());
  } else {
    return result;
  }
}

TensorValue convertRelu(OpBuilder &builder, Location loc, TensorValue x) {
  auto zero = builder.create<ConstantOp>(loc, 0.f);
  auto xType = x.getType().getDtype();
  bool cast = false;

  TensorValue xValue;
  if (x.getType().getDtype() != zero.getType().getDtype()) {
    // Cast X to fp32, then cast back
    xValue = builder.create<CastOp>(loc, x, zero.getType().getDtype());
    cast = true;
  } else {
    xValue = x;
  }

  auto broadcastZero = builder.create<BroadcastToOp>(loc, zero, xValue);
  auto max = builder.create<MaxOp>(loc, xValue, broadcastZero);
  if (cast) {
    return builder.create<CastOp>(loc, max, xType);
  } else {
    return max;
  }
}

TensorValue convertRelu6(OpBuilder &builder, Location loc, TensorValue x) {
  auto six = builder.create<ConstantOp>(loc, 6.f);
  auto zero = builder.create<ConstantOp>(loc, 0.f);
  TensorValue sixValue, zeroValue;
  if (x.getType().getDtype() != six.getType().getDtype()) {
    sixValue = builder.create<CastOp>(loc, six, x.getType().getDtype());
    zeroValue = builder.create<CastOp>(loc, zero, x.getType().getDtype());
  } else {
    sixValue = six;
    zeroValue = zero;
  }
  auto broadcastSix = builder.create<BroadcastToOp>(loc, sixValue, x);
  auto broadcastZero = builder.create<BroadcastToOp>(loc, zeroValue, x);
  auto min = builder.create<MinOp>(loc, x, broadcastSix);
  return builder.create<MaxOp>(loc, broadcastZero, min);
}

} // namespace M::MO
