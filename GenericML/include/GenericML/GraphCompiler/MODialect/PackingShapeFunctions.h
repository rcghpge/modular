//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "Support/MDialect/MAttrs.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace M::MO {

/// Materialize the shape logic for
///     `mo.invoke_shape_func["pack_matmul_b_shape_func"]`
std::optional<llvm::SmallVector<int64_t>>
materializePackMatmulShapeFunc(M::MO::InvokeShapeFuncOp invoke,
                               M::TargetInfoAttr targetAttr);

/// Materialize the shape logic for
///     `mo.invoke_shape_func["pack_filter_shape_func"]`
std::optional<llvm::SmallVector<int64_t>>
materializePackConvFilterShape(M::MO::InvokeShapeFuncOp invoke,
                               M::TargetInfoAttr targetAttr);

} // namespace M::MO
