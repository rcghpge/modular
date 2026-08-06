//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Lowers `map` dialect kernels into `mogg` codegen ops. Each `map` op that can
// appear inside (or as) a kernel implements `LowerableToMOGGInterface`'s
// `lowerToMOGG(MAPToMOGGState &)` -- see `MAPOpsLowerableToMOGG.cpp` for the
// per-op implementations. `MAPToMOGGState::lowerOp`/`lowerRegionBody` walk
// each `map.iter.opaque` and everything nested inside it, dispatching purely
// on whatever op the walk finds (`map.call.opaque` or
// `map.iter.foreach.body`) -- there is no separate opaque-vs-foreach branch.
//
// A `map.iter.opaque` becomes a `mogg._kernel`: its `!mo.tensor` inputs stay
// operands and its outputs stay results, but inside the body each tensor is
// reconstructed from a `!mogg.pointer_` block argument via
// `mogg._shape.create` / `mogg._tensor.create`, and a nested
// `map.call.opaque` becomes a `mogg._call.opaque` over those tensors, or a
// nested `map.iter.foreach.body` becomes a `mogg._call.foreach` driving the
// cloned per-index computation.
//
// A fused (store-epilogue) output -- a `map.iter.epilogue` feeding a call's
// outputs -- becomes a `mogg._tensor.create.epilogue` holding its per-index
// computation directly in its own body, recursed into by the same
// dispatcher (so a multi-op epilogue, e.g. compute-then-store, lowers just
// like any other body).
//
// Dynamic shapes are handled: the kernel gains one `!mogg.dim_` block
// argument per unique dynamic dimension (deduped across inputs, outputs, and
// any nested op's own shape attribute, e.g. `map.iter.foreach.body`'s),
// placed between the pointers and the context, and each `mogg._shape.create`
// is fed the dims for its own tensor.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Conversion/MAPToMOGG/MAPToMOGG.h"
#include "GraphCompiler/Interfaces/MAPToMOGGState.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOSHDialect/MOSHDialect.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_MAPTOMOGG
#include "GraphCompiler/Conversion/MAPToMOGG/MAPToMOGG.h.inc"

namespace {

struct MAPToMOGGPass : public MAP::impl::MAPToMOGGBase<MAPToMOGGPass> {
  using Base::Base;

  void runOnOperation() override {
    MOGG::MojoRegistry registry(getOperation());

    SmallVector<MAP::IterOpaqueOp> worklist;
    getOperation()->walk([&](MAP::IterOpaqueOp op) {
      worklist.push_back(op);
      // No need to recurse inside a `map.iter.opaque`.
      return WalkResult::skip();
    });

    for (MAP::IterOpaqueOp op : worklist) {
      MAPToMOGGState state(registry, op);
      if (failed(state.lowerOp(op.getOperation())))
        return signalPassFailure();
    }
  }
};

} // namespace

} // namespace M::MAP
