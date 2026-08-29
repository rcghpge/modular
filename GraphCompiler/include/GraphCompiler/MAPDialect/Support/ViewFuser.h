//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Fuses chains of `map.iter.foreach` kernels where at least one link is a
// view (any op whose body computes a load index through an
// `IndexTransformOpInterface` chain -- today `map.index.slice`/
// `map.index.broadcast`/`map.index.transpose`), generic over which transform
// kind is involved. Also fuses a view directly into an existing
// `map.iter.prologue` of a `map.iter.opaque` consumer, the one edge
// `PrologueFuser`'s own simpler mechanism can't take (it splices a
// producer's body in verbatim, which a transform op's `ParentOneOf` rejects
// inside a prologue).
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_VIEWFUSER_H
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_VIEWFUSER_H

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/Analysis/SequenceFusionAnalysis.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"

namespace M::MAP {

/// Fuses chains of `map.iter.foreach` ops within a `mo.graph` into single
/// kernels wherever at least one link is a view, splicing each producer's
/// body in at its own load site(s) and using that load's own (possibly
/// index-transformed) index -- unlike `ElementwiseFuser`, this does not
/// assume every op in the chain shares one index space, which is what lets a
/// view fuse into (or out of) a consumer of a different shape (e.g. a slice
/// feeding a transpose). A chain's sink may also be a `map.iter.opaque`
/// kernel, in which case the view fuses into whichever of its existing
/// `map.iter.prologue`s already captures it.
class ViewFuser {
public:
  /// Collects every fusible set of kernels in `graphOp`.
  explicit ViewFuser(MO::GraphOp graphOp);

  /// Fuses each set found at construction time into a single kernel, erases
  /// the fused-away kernels once they have no remaining users, and returns
  /// how many kernels were fused.
  unsigned fuse();

private:
  // Runs the source-destination sequence fusion analysis over `graphOp`,
  // finding every fusible set of kernels (eligible when at least one link is
  // a view) and computing a topological sort for each. Read-only: does not
  // mutate the graph.
  MOGG::FusionSubGraphs collectCandidates();

  // Fuses every op in `subgraph` but the last directly into the last one
  // (the subgraph's sink, either a `map.iter.foreach` or a
  // `map.iter.opaque`), which absorbs all of them.
  void fuseSubgraphOnce(llvm::ArrayRef<mlir::Operation *> subgraph);

  MO::GraphOp graphOp;
  MOGG::FusionSubGraphs subgraphs;
};

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_VIEWFUSER_H
