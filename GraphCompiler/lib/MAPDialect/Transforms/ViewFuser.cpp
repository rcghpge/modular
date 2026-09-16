//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Input-fusion wrapper for a view producer (a foreach whose output flows from a
// transformed load) fusing into a consumer foreach body or an existing
// `map.iter.prologue` capture of a `map.iter.opaque`. A pure elementwise
// producer is `ElementwiseFuser`'s job.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Support/InputFusion.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"

#include "mlir/Support/LLVM.h"

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEVIEWS
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

struct FuseViewsPass : public MAP::impl::FuseViewsBase<FuseViewsPass> {
  using Base::Base;

  void runOnOperation() override {
    InputFusionConfig config;
    // A view producer, decided from its store's dataflow. The consumer is a
    // foreach or an opaque -- the only kinds generic input fusion offers.
    config.shouldFuse = [](StoreOp srcStore, LoadOp) {
      return producesThroughView(srcStore);
    };
    // A view is a cheap index computation, always fine to re-derive per
    // consumer.
    config.canDuplicate = [](IterForeachOp) { return true; };
    numKernelsFused = fuseInput(getOperation(), config);
  }
};

} // namespace

} // namespace M::MAP
