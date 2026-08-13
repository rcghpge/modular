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

#include "GraphCompiler/MODialect/MOOps.h"

namespace M::MAP {

/// Folds a single-use elementwise `map.iter.foreach` consumer into the
/// `map.iter.epilogue` callback of the `map.iter.opaque` kernel that produces
/// its fused operand, so e.g. `matmul` followed by a bias `add` becomes one
/// kernel whose epilogue computes the add inline, per element, before storing.
class EpilogueFuser {
public:
  explicit EpilogueFuser(MO::GraphOp graphOp) : graphOp(graphOp) {}

  /// Repeatedly folds every eligible elementwise consumer into its sole
  /// producer's epilogue until no more fusions are possible, so a chain
  /// (e.g. matmul -> add -> relu) fully collapses, not just its first link.
  void fuse();

private:
  MO::GraphOp graphOp;
};

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_EPILOGUEFUSER_H
