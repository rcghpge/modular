//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Patterns/Algebraic.h"
#include "GraphCompiler/MODialect/Patterns/ShapeOpSinking.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/MOGGDialect/Support/SplitSlicePatterns.h"

#define DEBUG_TYPE "symbolic-optimization"

using namespace M;
using namespace MO;

namespace M::MO {
#define GEN_PASS_DEF_SYMBOLICOPTIMIZATION
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

namespace {

struct SymbolicOptimizationPass
    : public MO::impl::SymbolicOptimizationBase<SymbolicOptimizationPass> {
  using SymbolicOptimizationBase::SymbolicOptimizationBase;

  // Runs all default patterns plus the ones provided in the template
  // parameters.
  template <typename... PatternTypes>
  LogicalResult applyCorePatternsPlusExtraPatterns() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    if constexpr (sizeof...(PatternTypes)) {
      patterns.add<PatternTypes...>(context);
    }

    populateAlgebraicPatterns(patterns, symbolicFfastMath,
                              symbolicNoUndefinedMath);
    populateShapeOpSinkingPatterns(patterns);

    // TODO(24646): Ensure these use device aware rewriter.
    TransposeOp::getCanonicalizationPatterns(patterns, context);
    Static::BroadcastToOp::getCanonicalizationPatterns(patterns, context);

    // TODO(GEX-540): This is done to avoid potential subtle high-iteration
    // count loops in the pattern.
    mlir::GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    return applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }

  LogicalResult applyForwardMathematicalCanonicalizations() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateForwardCanonicalizationPatterns(patterns);

    // TODO(GEX-540): This is done to avoid potential subtle high-iteration
    // count loops in the pattern.
    mlir::GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    return applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }

  LogicalResult applyBackwardMathematicalCanonicalizations() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateBackwardCanonicalizationPatterns(patterns);

    // TODO(GEX-540): This is done to avoid potential subtle high-iteration
    // count loops in the pattern.
    mlir::GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    return applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }

  LogicalResult applyCorePatterns() {
    // First run all the optimizations with slices turned into split when
    // possible.
    if (applyCorePatternsPlusExtraPatterns<SliceToSplitPattern<>>().failed())
      return failure();

    // Once all split operation patterns have run, convert them to slice and run
    // all of the other patterns too. Without it, split patterns would never
    // apply because they would be turned into slices too early.
    return applyCorePatternsPlusExtraPatterns<SplitToSlicePattern<true>>();
  }

  void runOnOperation() override {
    if (applyCorePatterns().failed())
      return signalPassFailure();
    if (applyForwardMathematicalCanonicalizations().failed())
      return signalPassFailure();
    if (applyCorePatterns().failed())
      return signalPassFailure();
    if (applyBackwardMathematicalCanonicalizations().failed())
      return signalPassFailure();
    if (applyCorePatterns().failed())
      return signalPassFailure();
  }
};

} // namespace
