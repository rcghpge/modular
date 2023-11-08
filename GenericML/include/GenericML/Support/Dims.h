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

/// Normalize each dim in `dims` to the positive dim equivalent if it is
/// negative.
inline void normalizeNegativeDims(llvm::SmallVector<int64_t> &dims,
                                  size_t rank) {
  for (auto [i, dim] : llvm::enumerate(dims))
    dims[i] = normalizeAxis(dim, rank);
}

} // namespace M
