//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Functions for dim manipulation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace M {

inline int64_t normalizeAxis(int64_t axis, int64_t rank) {
  return (axis < 0) ? rank + axis : axis;
}

/// Clamp axis to the range [0, rank].
static int64_t clampAxis(int64_t axis, int64_t rank) {
  int64_t normalizedAxis = normalizeAxis(axis, rank);
  return std::max(std::min(normalizedAxis, rank), (int64_t)0);
}

/// Normalize each dim in `dims` to the positive dim equivalent if it is
/// negative.
inline void normalizeNegativeDims(llvm::SmallVector<int64_t> &dims,
                                  size_t rank) {
  for (auto [i, dim] : llvm::enumerate(dims))
    dims[i] = normalizeAxis(dim, rank);
}

} // namespace M
