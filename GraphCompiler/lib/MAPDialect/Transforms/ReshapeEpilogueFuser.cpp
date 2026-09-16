//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Output-fusion wrapper for a consumer that reads the producer's result through
// a single `map.index.reshape` of the loop index (and does more than relabel),
// collapsing e.g. `matmul -> reshape -> add` into one kernel by moving the
// reshape onto the epilogue's store index. An untransformed read is
// `EpilogueFuser`'s job; a bare reshape relabel is left as its own kernel.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Support/OutputFusion.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSERESHAPEEPILOGUE
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

struct FuseReshapeEpiloguePass
    : public MAP::impl::FuseReshapeEpilogueBase<FuseReshapeEpiloguePass> {
  using Base::Base;

  void runOnOperation() override {
    OutputFusionConfig config;
    config.shouldFuse = [](StoreOp, LoadOp consumerLoad) {
      return consumerReadsThroughReshape(consumerLoad) &&
             !consumerIsPureReshape(consumerLoad);
    };
    fuseOutput(getOperation(), config);
  }
};

} // namespace

} // namespace M::MAP
