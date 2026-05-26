//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/PatternMatch.h"

namespace M::MO {

/// Adds shape-op-sinking patterns: rewrites that re-shuffle the graph by
/// pushing reshape/broadcast/transpose/split/slice ops through neighboring
/// elementwise / reduction / matmul ops to expose more fusion and folding.
void populateShapeOpSinkingPatterns(mlir::RewritePatternSet &patterns);

} // namespace M::MO
