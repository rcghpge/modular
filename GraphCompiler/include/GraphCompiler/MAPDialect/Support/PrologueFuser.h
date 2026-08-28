//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Fuses a standalone elementwise `map.iter.foreach` producer directly into a
// `map.iter.opaque` consumer's `map.iter.prologue` callback on the input side
// it feeds.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_PROLOGUEFUSER_H
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_PROLOGUEFUSER_H

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "llvm/ADT/SmallVector.h"

namespace M::MAP {

/// Folds a sole-use elementwise `map.iter.foreach` producer into the
/// `map.iter.prologue` callback of the `map.iter.opaque` kernel whose input it
/// feeds, so e.g. `gather(add(a, b), indices)` becomes one kernel whose
/// data-operand prologue computes the add inline, per element, instead of
/// reading a separately materialized tensor.
class PrologueFuser {
public:
  /// Collects every prologue-fusible (dst, src) edge in `graphOp`.
  explicit PrologueFuser(MO::GraphOp graphOp);

  /// Folds every candidate found at construction time into its sole
  /// consumer's prologue and returns how many kernels it fused. A producer
  /// feeding two such operands is not duplicated: the sole-use requirement
  /// leaves it fused into neither.
  unsigned fuse();

private:
  struct FusionCandidate {
    IterOpaqueOp dst;
    IterPrologueOp prologueOp;
    unsigned captureIdx;
    IterForeachOp src;
    unsigned dstInputIdx;
  };

  // Finds every currently-eligible (prologueOp, captureIdx, src) edge in one
  // read-only pass over the graph. See PrologueFuser.cpp for the eligibility
  // rules.
  llvm::SmallVector<FusionCandidate> collectCandidates();

  MO::GraphOp graphOp;
  llvm::SmallVector<FusionCandidate> candidates;
};

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_PROLOGUEFUSER_H
