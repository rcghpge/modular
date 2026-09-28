//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Generic input fusion: inline a producer `map.iter.foreach` body at the load
// site(s) where a consumer reads its result, replacing the producer's store.
// The load site is either in a consumer foreach body or behind an existing
// `map.iter.prologue` capture of a `map.iter.opaque` consumer. The generic
// enforces every correctness constraint; a wrapper narrows which edges fuse
// with a `shouldFuse` predicate over the producer store and consumer load.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_INPUTFUSION_H
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_INPUTFUSION_H

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MOOps.h"

#include <cstdint>
#include <functional>

namespace M::MAP {

/// Whether `op`'s body contains any `IndexTransformOpInterface` op. A pure
/// elementwise kernel has none.
bool isViewForeach(IterForeachOp op);

/// Whether `store`'s value flows from a transformed load -- i.e. the kernel it
/// terminates is a view producer. Follows the actual store dataflow rather than
/// mere presence of a transform, and takes the store directly so it makes no
/// assumption about the producer having a single output.
bool producesThroughView(StoreOp store);

/// Whether `op` is a view and nothing else: it has an index transform but no
/// elementwise compute. Once a view fuses into an elementwise kernel the result
/// still reads through a transform, but it is no longer free to re-derive.
bool isPureViewForeach(IterForeachOp op);

/// Whether duplicating `op` into each of its consumers is not a net
/// memory-traffic loss: at most one of its inputs is a large tensor, since
/// every extra copy re-reads all of them. Mirrors MOGG's
/// FuseElementwiseKernels.cpp (JSC-30 / MSTDL-2759).
bool isCheapToDuplicate(IterForeachOp op);

/// The policy a wrapper supplies to narrow generic input fusion.
struct InputFusionConfig {
  /// Whether the edge from `producerStore` (a producer kernel's store) to
  /// `consumerLoad` (where the consumer reads that result) may fuse. The
  /// generic has already checked every correctness constraint.
  std::function<bool(StoreOp producerStore, LoadOp consumerLoad)> shouldFuse;
  /// Whether a producer with more than one use may be duplicated (the splice
  /// clones it into each consumer). A single-use producer always fuses.
  std::function<bool(IterForeachOp src)> canDuplicate;
  /// Cap on how many kernels may be fused into a single consumer, mirroring the
  /// legacy `kMaximumFusionSetSize`. Duplication lets a cheap multi-use
  /// producer be cloned into every consumer, so an unbounded elementwise chain
  /// with fan-out would otherwise grow a handful of enormous kernels. Draining
  /// stops once a consumer reaches this many fused kernel calls. Defaults to
  /// unbounded so callers that never duplicate (e.g. prologue fusion) are
  /// unaffected.
  uint64_t maxFusionSetSize = UINT64_MAX;
};

/// Greedily fuses producers into their consumers in `graphOp`, processing
/// consumers bottom-up and draining each consumer's producer chain. The stamped
/// `FusionInfo` follows the fusion itself: a prologue landing is `Prologue`, a
/// foreach landing `View` for a view producer else `Elementwise`. Returns how
/// many producers it fused.
unsigned fuseInput(MO::GraphOp graphOp, const InputFusionConfig &config);

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_INPUTFUSION_H
