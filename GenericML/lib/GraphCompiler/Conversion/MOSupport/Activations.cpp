//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOSupport/Activations.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"

namespace M::MO {

TensorValue convertPRelu(OpBuilder &builder, Location loc, TensorValue x,
                         TensorValue alpha) {
  auto zero = builder.create<ConstantOp>(loc, 0.f);
  auto broadcastZero = builder.create<BroadcastToOp>(loc, zero, x);
  auto broadcastAlpha = builder.create<BroadcastToOp>(loc, alpha, x);
  auto min = builder.create<MinOp>(loc, x, broadcastZero);
  auto minXAlpha = builder.create<MulOp>(loc, broadcastAlpha, min);
  auto max = builder.create<MaxOp>(loc, x, broadcastZero);
  return builder.create<AddOp>(loc, max, minXAlpha);
}

TensorValue convertRelu(OpBuilder &builder, Location loc, TensorValue x) {
  auto zero = builder.create<ConstantOp>(loc, 0.f);
  auto broadcastZero = builder.create<BroadcastToOp>(loc, zero, x);
  return builder.create<MaxOp>(loc, x, broadcastZero);
}

TensorValue convertRelu6(OpBuilder &builder, Location loc, TensorValue x) {
  auto six = builder.create<ConstantOp>(loc, 6.f);
  auto zero = builder.create<ConstantOp>(loc, 0.f);
  auto broadcastSix = builder.create<BroadcastToOp>(loc, six, x);
  auto broadcastZero = builder.create<BroadcastToOp>(loc, zero, x);
  auto min = builder.create<MinOp>(loc, x, broadcastSix);
  return builder.create<MaxOp>(loc, broadcastZero, min);
}

} // namespace M::MO
