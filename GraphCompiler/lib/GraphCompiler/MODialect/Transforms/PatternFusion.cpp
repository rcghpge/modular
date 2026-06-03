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
      // TODO(GEX-3695): Properly query the target info in the Graph Compiler.
      // Determine the effective arch name for the physical device. Or use
      // mockArchName if set (for testing only).
      StringRef effectiveArch = mockArchName.getValue();
      std::string queriedArch;
      if (effectiveArch.empty()) {
        // On macOS, Metal API requires main-thread access. MLIR runs passes
        // over multiple GraphOps in parallel (worker threads), so guard the
        // Metal device query with a main-thread check to prevent crashes. On
        // other platforms (e.g. Linux+CUDA), thread-safety is not an issue.
        bool canQueryDevice = true;
#if defined(__APPLE__)
        canQueryDevice = pthread_main_np() != 0;
#endif
        if (canQueryDevice) {
          auto gpuDevice = M::Driver::Device::create("gpu", 0);
          if (!gpuDevice.isError()) {
            auto archName = gpuDevice.get()->getArchName();
            if (!archName.isError()) {
              queriedArch = archName.get();
              effectiveArch = queriedArch;
            }
          }
        }
      }

      RewritePatternSet splitToSlicePatterns{ctx};
      splitToSlicePatterns.add<SplitToSlicePattern<>>(ctx);
      // MatmulAddPattern produces mo.composite.matmul_add which is only lowered
      // on SM100 (B200). Match any SM100 variant (e.g. sm_100a).
      if (effectiveArch.starts_with("sm_100"))
        MO::populateFusionMatmulAddPatterns(splitToSlicePatterns);
      if (failed(
              applyPatternsGreedily(graphOp, std::move(splitToSlicePatterns))))
        return signalPassFailure();
    }

    MO::synchronizeMetadataUpdates(graphOp);
  }
};

} // namespace

} // namespace M::MOGG
