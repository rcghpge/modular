//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_GRAPH_COMPILER_MOGGDIALECT_UTILS_H
#define GENERICML_GRAPH_COMPILER_MOGGDIALECT_UTILS_H

#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"

using namespace M;

namespace M::MOGG {

/// Inline the given lambda into the call site of the invoke and return the new
/// results which have replaced the old operation.
SmallVector<Value> inlineInvoke(MOGG::LambdaInvokeOp invokeOp,
                                MOGG::LambdaOp invokeTarget);

/// Create a call to the reshape kernel directly in MOGG.
MOGG::CallOp createMOGGReshape(mlir::OpBuilder &builder, mlir::Location loc,
                               Value tensorIn, MO::TensorType newShape);

// Create a call to the slice kernel directly in MOGG.
MOGG::CallOp createMOGGSlice(mlir::OpBuilder &builder, mlir::Location loc,
                             Value tensorIn, Type outType,
                             const SmallVector<int64_t> &start,
                             const SmallVector<int64_t> &stop,
                             const SmallVector<int64_t> &step);

} // namespace M::MOGG

#endif // GENERICML_GRAPH_COMPILER_MOGGDIALECT_UTILS_H
