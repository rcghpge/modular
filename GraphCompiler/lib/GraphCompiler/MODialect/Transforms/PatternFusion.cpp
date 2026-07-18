//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/GraphCompiler/MODialect/Patterns/Fusion.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/SplitSlicePatterns.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#if defined(__APPLE__)
#include <pthread.h>
#endif

using namespace M;
using namespace M::MO;

namespace M::MOGG {

#define GEN_PASS_DEF_PATTERNFUSION
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct PatternFusionPass : public impl::PatternFusionBase<PatternFusionPass> {
  using PatternFusionBase::PatternFusionBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    M::DeviceRefAttr targetDeviceRef =
        targetDevice == "" ? nullptr
                           : M::DeviceRefAttr::get(ctx, targetDevice.getValue(),
                                                   targetDeviceId.getValue());

    // Apply the special pattern to fuse allReduceOps at the beginning.
    llvm::SmallVector<MO::AllReduceSumOp> allReduceOps;
    getOperation()->walk([&](MO::AllReduceSumOp allReduceSumOp) {
      allReduceOps.push_back(allReduceSumOp);
    });
    for (auto allReduceOp : allReduceOps)
      MO::applyFuseAllReduceAddRMSNormQuantFP8(allReduceOp, targetDeviceRef);

    // Parallel-region analog: fuse `mo.parallel { mo.bundled.allreduce.sum }`
    // followed by per-device add + rms_norm + quantize_fp8 chains outside the
    // parallel into a single `mo.parallel { fused }` op in one step.
    llvm::SmallVector<MO::BundledAllReduceSumOp> bundledAllReduceOps;
    getOperation()->walk([&](MO::BundledAllReduceSumOp op) {
      bundledAllReduceOps.push_back(op);
    });
    for (auto bundledAllReduceOp : bundledAllReduceOps) {
      // Earlier rewrites in this same loop may have erased the parent
      // parallel of this op; skip stale entries.
      if (!bundledAllReduceOp.getOperation()->getBlock())
        continue;
      MO::applyFuseParallelBundledAllReduceAddRMSNormQuantFP8(
          bundledAllReduceOp);
    }

    MO::GraphOp graphOp = getOperation();

    // Phase 1: Convert slices into splits so that the RoPE pattern can match
    // split ops created from adjacent slice pairs.
    {
      RewritePatternSet sliceToSplitPatterns{ctx};
      sliceToSplitPatterns.add<SliceToSplitPattern</*AllowDynamicStop=*/true>>(
          ctx);
      if (failed(
              applyPatternsGreedily(graphOp, std::move(sliceToSplitPatterns))))
        return signalPassFailure();
    }

    // Phase 2: Main fusion patterns including the new RoPE fusion.
    MO::PackParamCache cache;
    MO::MHAOptions mhaOpts{
        enableMHAPattern.getValue(),
        mhaAllowUnsafeBCast.getValue(),
        mhaAllowInternalReshape.getValue(),
        debugMHAPattern.getValue(),
    };

    RewritePatternSet patterns{ctx};
    MO::populateFusionPatterns(patterns, cache, targetDeviceRef, mhaOpts);

    if (failed(applyPatternsGreedily(graphOp, std::move(patterns))))
      return signalPassFailure();

    // Phase 3: Lower any splits that survived fusion back to slices.
    {
      MO::DeviceInfoMapping deviceInfoMapping(
          graphOp->getParentOfType<mlir::ModuleOp>());
      RewritePatternSet splitToSlicePatterns{ctx};
      splitToSlicePatterns.add<SplitToSlicePattern<>>(ctx);
      MO::populateFusionMatmulAddPatterns(deviceInfoMapping,
                                          splitToSlicePatterns);
      // MegaFFNNvfp4Pattern produces mo.composite.mega_ffn_nvfp4, whose kernel
      // is MAX-only and SM100-only. The pattern arch-gates itself via
      // deviceInfoMapping; it is additionally behind a pass option so OSS
      // builds (where the fused kernel is unregistered) can disable it.
      if (enableMegaFFNPattern.getValue())
        MO::populateFusionMegaFFNPatterns(deviceInfoMapping,
                                          splitToSlicePatterns);
      if (failed(
              applyPatternsGreedily(graphOp, std::move(splitToSlicePatterns))))
        return signalPassFailure();
    }

    MO::synchronizeMetadataUpdates(graphOp);
  }
};

} // namespace

} // namespace M::MOGG
