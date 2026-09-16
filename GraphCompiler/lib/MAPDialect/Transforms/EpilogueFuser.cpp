//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Output-fusion wrapper for a consumer that reads the producer's result at an
// untransformed index, so e.g. `matmul` followed by a bias `add` becomes one
// kernel whose epilogue computes the add inline. A reshape read is
// `ReshapeEpilogueFuser`'s job.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Support/OutputFusion.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEEPILOGUE
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

struct FuseEpiloguePass : public MAP::impl::FuseEpilogueBase<FuseEpiloguePass> {
  using Base::Base;
  FuseEpiloguePass(const FuseEpiloguePass &other) : Base(other) {}

  void runOnOperation() override {
    OutputFusionConfig config;
    config.shouldFuse = [](StoreOp, LoadOp consumerLoad) {
      return consumerReadsUntransformed(consumerLoad);
    };
    numKernelsFused = fuseOutput(getOperation(), config);
  }

  Statistic numKernelsFused{this, "num-kernels-fused",
                            "Number of kernels fused"};
};

} // namespace

} // namespace M::MAP
