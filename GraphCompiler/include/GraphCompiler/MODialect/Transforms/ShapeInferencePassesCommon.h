//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_SHAPEINFERENCEPASSES_COMMON_H
#define GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_SHAPEINFERENCEPASSES_COMMON_H

#include "GraphCompiler/MODialect/MODevices.h"

using namespace M;
using namespace MO;
namespace M::MO::ShapeInferencePassCommon {

//===----------------------------------------------------------------------===//
// Common Patterns
//===----------------------------------------------------------------------===//

/// A pattern that folds/flattens a chain of rebinds into a tree of rebinds that
/// share the same input.
///
/// It serves 2 purposes:
/// 1. Get rid of intermediate rebinds to unknown shapes (see example below).
/// 2. Minimizes the number of rebinds we must look through when checking for
/// constants.
///
/// Example:
///
/// input : tensor<[D0]>
///
///   reb1 = rebind(input) : tensor<[D0]> -> tensor<[?]>
///   res  = rebind(reb1)  : tensor<[?]>  -> tensor<[D2]>
/// ===>
///   reb1 = rebind(input) : tensor<[D0]> -> tensor<[?]>
///   res  = rebind(input) : tensor<[D0]> -> tensor<[D2]>
class FoldConsecutiveRebinds
    : public DeviceAwareRewritePattern<OpRewritePattern, RebindOp> {
  using DeviceAwareRewritePattern::DeviceAwareRewritePattern;

  LogicalResult
  deviceAwareMatchAndRewrite(RebindOp op,
                             DeviceAwareRewriter &rewriter) const override;
};

} // namespace M::MO::ShapeInferencePassCommon

#endif // GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_SHAPEINFERENCEPASSES_COMMON_H
