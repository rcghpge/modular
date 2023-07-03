//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Functions for dim manipulation.
//
//===----------------------------------------------------------------------===//

#include "Support/LLVMForwardDecls.h"

namespace M {

/// Normalize each dim in `dims` to the positive dim equivalent if it is
/// negative.
inline void normalizeNegativeDims(SmallVectorImpl<int64_t> &dims, size_t rank) {
  for (auto &dim : dims) {
    if (dim < 0)
      dim += rank;
  }
}

} // namespace M
