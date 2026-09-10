//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Fuses a reshape (folded by ElementwiseFuser into a `map.iter.foreach`
// consumer) into the `map.iter.epilogue` of the `map.iter.opaque` producer
// that feeds it, by relocating the reshape's index transform onto the store.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_RESHAPEEPILOGUEFUSER_H
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_RESHAPEEPILOGUEFUSER_H

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "llvm/ADT/SmallVector.h"

namespace M::MAP {

/// Folds a `map.iter.foreach` consumer that reads a single-use
/// `map.iter.opaque` producer's result through a `map.index.reshape` into that
/// producer's `map.iter.epilogue`, by moving the reshape (with its signature's
/// in/out shapes swapped -- the store direction is the inverse of the load
/// direction) onto the epilogue's store index. This re-exposes any downstream
/// elementwise op as a plain untransformed consumer, which `EpilogueFuser` can
/// then absorb too, collapsing e.g. `matmul -> reshape -> add` into one kernel.
///
/// ElementwiseFuser runs first and folds a standalone reshape into its
/// consumer, so this fuser sees the reshape embedded in a consumer body rather
/// than as its own kernel. A pure reshape (no other per-element compute) is
/// left alone: it is a no-op relabel in MGP, so relocating it into an epilogue
/// would only add per-element index math for no benefit.
class ReshapeEpilogueFuser {
public:
  /// Collects every reshape-epilogue-fusible (dst, src) edge in `graphOp`.
  explicit ReshapeEpilogueFuser(MO::GraphOp graphOp);

  /// Fuses every candidate found at construction time and returns how many
  /// kernels it fused. Applies exactly one round.
  unsigned fuse();

private:
  struct FusionCandidate {
    IterOpaqueOp dst;
    unsigned resultIdx;
    IterEpilogueOp epilogueOp;
    IterForeachOp src;
    // The `map.index.reshape` inside `src` that transforms `src`'s loop index
    // into the load index of `dst`'s (fused) result.
    ReshapeIndexOp reshape;
  };

  // Finds every currently-eligible (dst, src) edge in one read-only pass over
  // the graph. See ReshapeEpilogueFuser.cpp for the eligibility rules.
  llvm::SmallVector<FusionCandidate> collectCandidates();

  MO::GraphOp graphOp;
  llvm::SmallVector<FusionCandidate> candidates;
};

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_RESHAPEEPILOGUEFUSER_H
