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

#include "GraphCompiler/MODialect/MOOps.h"

namespace M::MAP {

/// Folds a sole-use elementwise `map.iter.foreach` producer into the
/// `map.iter.prologue` callback of the `map.iter.opaque` kernel whose input it
/// feeds, so e.g. `gather(add(a, b), indices)` becomes one kernel whose
/// data-operand prologue computes the add inline, per element, instead of
/// reading a separately materialized tensor.
class PrologueFuser {
public:
  explicit PrologueFuser(MO::GraphOp graphOp) : graphOp(graphOp) {}

  /// Folds every eligible elementwise producer into its sole consumer's
  /// prologue. A producer feeding two such operands is not duplicated: the
  /// sole-use requirement leaves it fused into neither.
  void fuse();

private:
  MO::GraphOp graphOp;
};

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_PROLOGUEFUSER_H
