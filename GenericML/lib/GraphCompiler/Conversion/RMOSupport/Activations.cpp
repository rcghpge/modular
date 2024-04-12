//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/RMOSupport/Activations.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/RMODialect/RMOOps.h"

using M::MO::TensorValue;

namespace M::RMO {

TensorValue convertPRelu(OpBuilder &builder, Location loc, TensorValue x,
                         TensorValue alpha) {
  auto zero = builder.create<MO::ConstantOp>(loc, 0.f);

  TensorValue xValue;
  bool cast = false;
  if (x.getType().getDtype() != zero.getType().getDtype()) {
    cast = true;
    xValue = builder.create<RMO::MOCastOp>(loc, x, zero.getType().getDtype());
  } else {
    xValue = x;
  }

  auto broadcastZero = builder.create<RMO::BroadcastLikeOp>(loc, zero, xValue);
  auto broadcastAlpha =
      builder.create<RMO::BroadcastLikeOp>(loc, alpha, xValue);
  auto min = builder.create<RMO::MOMinOp>(loc, xValue.getType(), xValue,
                                          broadcastZero);
  auto minXAlpha = builder.create<RMO::MOMulOp>(loc, broadcastAlpha.getType(),
                                                broadcastAlpha, min);
  auto max = builder.create<RMO::MOMaxOp>(loc, xValue.getType(), xValue,
                                          broadcastZero);

  auto result =
      builder.create<RMO::MOAddOp>(loc, max.getType(), max, minXAlpha);
  if (cast) {
    return builder.create<RMO::MOCastOp>(loc, result, x.getType().getDtype());
  } else {
    return result;
  }
}

TensorValue convertRelu(OpBuilder &builder, Location loc, TensorValue x) {
  auto zero = builder.create<MO::ConstantOp>(loc, 0.f);
  auto xType = x.getType().getDtype();
  bool cast = false;

  TensorValue xValue;
  if (x.getType().getDtype() != zero.getType().getDtype()) {
    // Cast X to fp32, then cast back
    xValue = builder.create<RMO::MOCastOp>(loc, x, zero.getType().getDtype());
    cast = true;
  } else {
    xValue = x;
  }

  auto broadcastZero = builder.create<RMO::BroadcastLikeOp>(loc, zero, xValue);
  auto max = builder.create<RMO::MOMaxOp>(loc, xValue.getType(), xValue,
                                          broadcastZero);
  if (cast) {
    return builder.create<RMO::MOCastOp>(loc, max, xType);
  } else {
    return max;
  }
}

TensorValue convertRelu6(OpBuilder &builder, Location loc, TensorValue x) {
  auto six = builder.create<MO::ConstantOp>(loc, 6.f);
  auto zero = builder.create<MO::ConstantOp>(loc, 0.f);
  TensorValue sixValue, zeroValue;
  if (x.getType().getDtype() != six.getType().getDtype()) {
    sixValue = builder.create<RMO::MOCastOp>(loc, six, x.getType().getDtype());
    zeroValue =
        builder.create<RMO::MOCastOp>(loc, zero, x.getType().getDtype());
  } else {
    sixValue = six;
    zeroValue = zero;
  }
  auto broadcastSix = builder.create<RMO::BroadcastLikeOp>(loc, sixValue, x);
  auto broadcastZero = builder.create<RMO::BroadcastLikeOp>(loc, zeroValue, x);
  auto min = builder.create<MinOp>(loc, x, broadcastSix);
  return builder.create<MaxOp>(loc, broadcastZero, min);
}

} // namespace M::RMO
