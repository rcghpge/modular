//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/MGPTransforms.h"

#include "./AsyncDataflowDependencies.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPTransforms/Utils.h"
#include "MLRT/GraphRT/GRTDialect/GRTChainInterface.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "MLRT/GraphRT/MEFDialect/MEFTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

#include "Support/Compiler/Threading.h"

#include <queue>

using namespace mlir;

#define DEBUG_TYPE "mgp-simplify-chains-pass"

namespace M::MGP {

// Generate support code for SimplifyChainsPass
#define GEN_PASS_DEF_SIMPLIFYCHAINSPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// SimplifyChainsPass
//===----------------------------------------------------------------------===//

class SimplifyChainsPass
    : public impl::SimplifyChainsPassBase<SimplifyChainsPass> {
  using SimplifyChainsPassBase::SimplifyChainsPassBase;

  static ValueRange getGatingOperands(Operation *op) {
    if (!op)
      return ValueRange{};

    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op))
      // Only the extra strict operands are gating for control flow ops.
      return cfOp.getStrictExtraOperands();

    if (isa<MEF::NonStrictInterface>(op))
      return ValueRange{};
    return op->getOperands();
  }

  static SmallVector<Value> accumulatePrimitiveChains(Value chain) {
    SmallVector<Value> result;

    std::queue<Value> stack;
    stack.push(chain);

    llvm::SmallDenseSet<Value> visited;

    // Depth-first walk through grt.chain.create ops.
    while (!stack.empty()) {
      Value top = stack.front();
      stack.pop();

      if (!visited.insert(top).second)
        continue;

      // Walk through chain create ops.
      if (auto chain = top.getDefiningOp<GRT::ChainCreateOp>()) {
        for (auto input : chain.getInputChains())
          stack.push(input);

        continue;
      }

      result.push_back(top);
    }

    return result;
  }

  /// Eliminate inputs to `grt.chain.create` operations that are implied ready
  /// by other inputs. Given a chain create op:
  ///
  /// %X = grt.chain.create(%0, %1, %2)
  ///
  /// The algorithm works by walking through the transitive inputs of
  /// the produces of %0, %1, and %2 looking for the values %0, %1, and %2.
  /// The query this search is answering is:
  ///
  /// * is %0 reachable from %1,
  /// * is %1 reachable from %2,
  /// * ...
  ///
  /// i.e. is any input to the `grt.chain.create` reachable from any other
  /// operands. Those that are reachable are deleted.
  SmallVector<Value, 16>
  eliminateImpliedByOperands(GRT::ChainCreateOp chainCreate) {
    // Maintain an association between the values we are interested in and
    // a bool indicating whether they were found while walking the implied-by
    // relation.
    llvm::SmallSetVector<Value, 16> needles;

    // Initialize all values we're searching for to false.
    for (auto arg : chainCreate.getInputChains()) {
      for (auto needle : accumulatePrimitiveChains(arg))
        needles.insert(needle);
    }

    llvm::SmallDenseSet<Value> visited;
    llvm::SetVector<Value> currentGeneration;
    llvm::SetVector<Value> newGeneration;

    // Seed the search with the immediate implied-by set of needles.
    for (auto needle : needles)
      currentGeneration.insert_range(getGatingOperands(needle.getDefiningOp()));

    // Iteratively walk the implied by relation, looking for any of the needles.
    while (!currentGeneration.empty()) {
      // Mark the current generation as visited
      visited.insert_range(currentGeneration);

      // See if any value in the current generation is one of the needles we're
      // searching for. If a needle is found, delete it from the needle set.
      needles.remove_if([&](Value v) { return currentGeneration.contains(v); });

      newGeneration.clear();

      // Determine the next generation by following the implied-by relation one
      // step.
      for (auto val : currentGeneration) {
        for (auto next : getGatingOperands(val.getDefiningOp())) {
          if (!visited.contains(next))
            newGeneration.insert(next);
        }
      }

      std::swap(newGeneration, currentGeneration);
    }

    return needles.takeVector();
  }

  void runOnOperation() override {
    ModelOp modelOp = getOperation();

    SmallVector<GRT::ChainCreateOp> chainCreates;

    modelOp.walk([&](GRT::ChainCreateOp op) { chainCreates.push_back(op); });

    using CacheT =
        SmallVector<std::pair<GRT::ChainCreateOp, SmallVector<Value>>>;

    auto consolidate = [](CacheT &source, MutableArrayRef<CacheT> others) {
      for (CacheT &other : others)
        source.append(std::move(other));
    };

    auto func = [&](CacheT &cache, GRT::ChainCreateOp op) {
      cache.emplace_back(op, eliminateImpliedByOperands(op));
      return success();
    };

    CacheT cache;
    cache.reserve(chainCreates.size());

    [[maybe_unused]] auto result = failableParallelForEach(
        &getContext(), chainCreates, func, cache, consolidate);

    ASSERT_STREAM(succeeded(result), "processing chain create ops failed");

    for (auto &[chainOp, values] : cache)
      chainOp.setInputChains(values);

    RewritePatternSet patterns(&getContext());
    GRT::ChainCreateOp::getCanonicalizationPatterns(patterns, &getContext());

    auto asOperations = llvm::to_vector_of<Operation *>(chainCreates);

    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
    (void)applyOpPatternsGreedily(asOperations, std::move(patterns), config);
  }
};

} // namespace

} // namespace M::MGP
