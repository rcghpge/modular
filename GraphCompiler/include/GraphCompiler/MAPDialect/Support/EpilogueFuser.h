//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Fuses an elementwise `map.iter.foreach` consumer directly into the sole
// `map.iter.opaque` producer that feeds it, by inlining the consumer's
// per-element compute into the producer's `map.iter.epilogue` callback.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_EPILOGUEFUSER_H
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_EPILOGUEFUSER_H

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "llvm/ADT/SmallVector.h"

namespace M::MAP {

/// Folds a single-use elementwise `map.iter.foreach` consumer into the
/// `map.iter.epilogue` callback of the `map.iter.opaque` kernel that produces
/// its fused operand, so e.g. `matmul` followed by a bias `add` becomes one
/// kernel whose epilogue computes the add inline, per element, before storing.
class EpilogueFuser {
public:
  /// Collects every epilogue-fusible (dst, src) edge in `graphOp`.
  explicit EpilogueFuser(MO::GraphOp graphOp);

  /// Fuses every candidate found at construction time and returns how many
  /// kernels it fused. Applies exactly one round: folding one link of a
  /// chain (e.g. matmul -> add -> relu) can expose the next, so a caller
  /// wanting the whole chain collapsed must reconstruct and re-fuse until a
  /// round returns zero.
  unsigned fuse();

private:
  struct FusionCandidate {
    IterOpaqueOp dst;
    unsigned resultIdx;
    IterEpilogueOp epilogueOp;
    IterForeachOp src;
  };

  // Finds every currently-eligible (dst, src) edge in one read-only pass
  // over the graph. See EpilogueFuser.cpp for the eligibility rules.
  llvm::SmallVector<FusionCandidate> collectCandidates();

  MO::GraphOp graphOp;
  llvm::SmallVector<FusionCandidate> candidates;
};

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_EPILOGUEFUSER_H
