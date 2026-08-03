//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Fuses elementwise `map.iter.foreach` kernels by inlining producers into
// consumers.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_ELEMENTWISEFUSER_H
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_ELEMENTWISEFUSER_H

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/Analysis/SequenceFusionAnalysis.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"

namespace M::MAP {

/// Fuses chains of elementwise `map.iter.foreach` ops within a `mo.graph` into
/// single `map.iter.foreach` kernels, inlining each producer's per-element
/// compute directly into its consumer's body.
class ElementwiseFuser {
public:
  // Runs the source-destination sequence fusion analysis over `graphOp`,
  // finding every fusible set of `map.iter.foreach` kernels and computing a
  // topological sort for each. Read-only: does not mutate the graph.
  explicit ElementwiseFuser(MO::GraphOp graphOp);

  /// Fuses each set found during construction into a single kernel, then
  /// erases the fused-away kernels once they have no remaining users.
  void fuse();

private:
  // Fuses every op in `subgraph` but the last directly into the last one
  // (the subgraph's sink), which absorbs all of them.
  void fuseSubgraph(llvm::ArrayRef<mlir::Operation *> subgraph);

  MO::GraphOp graphOp;
  MOGG::FusionSubGraphs subgraphs;
};

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_ELEMENTWISEFUSER_H
