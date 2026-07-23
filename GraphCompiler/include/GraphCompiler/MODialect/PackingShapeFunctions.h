//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOOps.h"
#include "Support/MDialect/MAttrs.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace M::MO {

/// Materialize the shape logic for
///     `mo.invoke_shape_func["pack_matmul_b_shape_func"]`
std::optional<llvm::SmallVector<int64_t>>
materializePackMatmulShapeFunc(InvokeShapeFuncOp invoke,
                               TargetInfoAttr targetAttr);

/// Materialize the shape logic for
///     `mo.invoke_shape_func["pack_filter_shape_func"]`
std::optional<llvm::SmallVector<int64_t>>
materializePackConvFilterShape(InvokeShapeFuncOp invoke,
                               TargetInfoAttr targetAttr);

std::optional<SmallVector<int64_t>>
materializePackConvTransposeFilterShape(InvokeShapeFuncOp invoke,
                                        TargetInfoAttr targetAttr);

} // namespace M::MO
