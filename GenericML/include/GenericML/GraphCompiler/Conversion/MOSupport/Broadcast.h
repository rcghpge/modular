//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOSUPPORT_BROADCAST_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOSUPPORT_BROADCAST_H

namespace M::MO {

/// Reshape the `from` tensor using an intermediate shape, then attempt to
/// broadcast to the shape of the `to` tensor.
/// Does not currently support reshaping to dynamic or parametric shapes.
MOTensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                    MOTensorValue from, TensorType reshapeType,
                                    MOTensorValue to);

/// Reshape the `from` tensor using an intermediate shape, then attempt to
/// broadcast to the shape of the `to` tensor.
/// Does not currently support reshaping to dynamic or parametric shapes.
MOTensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                    MOTensorValue from,
                                    ArrayRef<int64_t> reshapeShape,
                                    MOTensorValue to);

/// Broadcast tensor `x` and `y` to each other, such that they have the same
/// shape after broadcasting. The method broadcasts statically, if the input
/// shapes are static, or dynamically otherwise.
std::pair<MOTensorValue, MOTensorValue>
reifyImplicitBroadcasting(OpBuilder &builder, Location loc, MOTensorValue x,
                          MOTensorValue y);

/// Broadcast tensor `x` to `y`. Requires that `x` can be broadcast
/// unidirectionally to `y`.
MOTensorValue broadcastTo(OpBuilder &builder, Location loc, MOTensorValue x,
                          MOTensorValue y);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOSUPPORT_BORADCAST_H
