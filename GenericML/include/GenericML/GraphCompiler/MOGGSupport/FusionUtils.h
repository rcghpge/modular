//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_GRAPH_COMPILER_MOGGSUPPORT_FUSIONUTILS_H
#define GENERICML_GRAPH_COMPILER_MOGGSUPPORT_FUSIONUTILS_H

#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"

using namespace M;

namespace M::MOGG {

/// Returns a compact, human readable 'summary' of op, which can be used
/// at runtime for debugging and profiling.
StringAttr getSummaryString(Operation *op);

/// Returns a compact, human readable 'summary' of the combination of ops.
StringAttr getSummaryString(ArrayRef<Operation *> ops);

/// Inline the given lambda into the call site of the invoke and return the new
/// results which have replaced the old operation.
SmallVector<Value> inlineInvoke(MOGG::LambdaInvokeOp invokeOp,
                                MOGG::LambdaOp invokeTarget);

/// This inlines all lambdas which are only invoked within other lambdas
/// and removed any unused lambdas. Returns true if there was any
/// modification
bool inlineLambdasInKernel(KernelOp kernel);

/// Create a call to the reshape kernel directly in MOGG.
MOGG::CallOp createMOGGReshape(mlir::OpBuilder &builder, mlir::Location loc,
                               Value tensorIn, MO::TensorType newShape);

// Create a call to the slice kernel directly in MOGG.
MOGG::CallOp createMOGGSlice(mlir::OpBuilder &builder, mlir::Location loc,
                             Value tensorIn, Type outType,
                             const SmallVector<int64_t> &start,
                             const SmallVector<int64_t> &stop,
                             const SmallVector<int64_t> &step);

/// Fuse small constant inputs to `kernel` into the kernel, and erase the
/// original constant op if it has no remaining users.
///
/// Example:
///   %const = mo.constant ...
///   %res   = mogg.kernel(%const, ...) {
///     ...
///   }
///
/// ===>
///
///   %res = mogg.kernel(...) {
///     %const = mo.constant ...
///     ...
///   }
void fuseConstant(MOGG::KernelOp kernel);

} // namespace M::MOGG

#endif // GENERICML_GRAPH_COMPILER_MOGGSUPPORT_FUSIONUTILS_H
