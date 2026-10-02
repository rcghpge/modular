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

/// Flattens a chain of rebinds into rebinds of the same input, dropping
/// intermediate rebinds to unknown shapes and shortening the chains constant
/// checks must look through.
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
