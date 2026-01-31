//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_STATICRESHAPEDIMENSIONSMAPPINGANALYSIS_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_STATICRESHAPEDIMENSIONSMAPPINGANALYSIS_H

#include "Support/AssertStream.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <optional>

namespace M::MO {

/// Static reshape analysis that will return the mapping between input and
/// output shapes across a reshape operation. Dynamic dimensions are ignored.
/// This will fail whenever the dynamic dimensions are mixed with static
/// dimensions.
/// EXAMPLE:
/// Works: D0, 4, 6, 1 -> D0, 2, 3, 2, 1, 2
///   Forward: [[], [1, 2], [2, 3, 5]]
///   Backward: [[], [1], [1, 2], [2], [2]]
/// Does not work: D0, 4, 6, 1 -> 4, D0, 6, 1
///
/// 1s are ignored because the mapping is not well defined (could be
/// mapped to any neighbor dimension). This means that their mapping will be
/// empty.
///
/// USECASE:
/// This analysis groups dimensions of a reshape by mapping dimensions that are
/// now "mixed" with one another. If a dimension in `before` maps to another in
/// `after`, it means that in the `after` shape, the mapped dimension originated
/// from the dimension in `before` (and vice versa, `before` contributed to
/// `after`). This helps us do things like change the split axis of a `mo.split`
/// across a `mo.reshape` and guarantee correctness because we know where to cut
/// and not impact other dimensions.
class StaticReshapeDimensionsMappingAnalysis {
public:
  // Before := shape before the reshape
  // After := shape after the reshape
  StaticReshapeDimensionsMappingAnalysis(llvm::ArrayRef<mlir::TypedAttr> before,
                                         llvm::ArrayRef<mlir::TypedAttr> after);
  ~StaticReshapeDimensionsMappingAnalysis() = default;

  // If the reshape is supported.
  inline operator bool() const { return isSupported; }

  using DimensionsMapping = llvm::SmallVector<llvm::SmallVector<int64_t>>;

  // Maps before dimensions to after dimensions.
  // Example:
  // Before  : [1, 2, 3, 4]
  // After   : [1, 2, 3, 2, 2]
  // Forward : [[], [1], [2], [3, 4]]
  inline const DimensionsMapping &forwardMapping() const {
    assert(isSupported);
    return forward;
  }

  // Maps after dimensions to before dimensions.
  // Example:
  // Before   : [1, 2, 3, 4]
  // After    : [1, 2, 3, 2, 2]
  // Backward : [[], [1], [2], [3], [3]]
  inline const DimensionsMapping &backwardMapping() const {
    assert(isSupported);
    return backward;
  }

private:
  DimensionsMapping forward;
  DimensionsMapping backward;
  bool isSupported;
};

} // namespace M::MO

#endif
