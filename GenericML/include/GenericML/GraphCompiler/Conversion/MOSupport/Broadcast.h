//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Broadcast helpers for frontends converting to MO.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOSUPPORT_BROADCAST_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOSUPPORT_BROADCAST_H

namespace M::MO {

/// Reshape the `from` tensor using an intermediate shape, then attempt to
/// broadcast to the shape of the `to` tensor.
/// Does not currently support reshaping to dynamic or parametric shapes.
TensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                  TensorValue from, TensorType reshapeType,
                                  TensorValue to);

/// Reshape the `from` tensor using an intermediate shape, then attempt to
/// broadcast to the shape of the `to` tensor.
/// Does not currently support reshaping to dynamic or parametric shapes.
TensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                  TensorValue from,
                                  ArrayRef<int64_t> reshapeShape,
                                  TensorValue to);

/// Reshape the `from` tensor using an intermediate shape (which can be
/// dynamic), then attempt to broadcast to the shape of the `to` tensor.
TensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                  TensorValue from, TensorType reshapeType,
                                  TensorValue reshapeShape, TensorValue to);

/// Reshape the `from` tensor using an intermediate shape (which can be
/// dynamic), then attempt to broadcast to the shape of the `to` tensor.
TensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                  TensorValue from,
                                  ArrayRef<int64_t> reshapeDims,
                                  TensorValue reshapeShape, TensorValue to);

/// Broadcast tensor `x` to the shape of `s`. Works with static shapes only.
TensorValue broadcastToStaticShape(OpBuilder &builder, Location loc,
                                   TensorValue x, ShapeAttr s);

/// Broadcast tensor `x` and `y` to each other, such that they have the same
/// shape after broadcasting. The method broadcasts statically, if the input
/// shapes are static, or dynamically otherwise.
/// All inputs must have a static rank and compatible shapes.
std::pair<TensorValue, TensorValue>
reifyImplicitBroadcasting(OpBuilder &builder, Location loc, TensorValue x,
                          TensorValue y);

/// Broadcast tensor `x`, `y` and `z` to each other, such that they have the
/// same shape after broadcasting. The method broadcasts statically, if the
/// input shapes are static, or dynamically otherwise.
/// All inputs must have a static rank and compatible shapes.
std::tuple<TensorValue, TensorValue, TensorValue>
reifyImplicitBroadcasting(OpBuilder &builder, Location loc, TensorValue x,
                          TensorValue y, TensorValue z);

/// Broadcast tensor `x` to `y`. Requires that `x` can be broadcast
/// unidirectionally to `y`.
TensorValue broadcastTo(OpBuilder &builder, Location loc, TensorValue x,
                        TensorValue y);

/// Broadcasts a float `scalar` to `toVal`'s shape.
/// Does so via a `ConstantOp` with the same float element type as `toVal`.
TensorValue broadcastFloatTo(OpBuilder &builder, Location loc,
                             const APFloat &scalar, TensorValue toVal);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOSUPPORT_BROADCAST_H
