//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// mo-merge-adjacent-parallels
//
// Peephole pass that fuses two adjacent `mo.parallel` regions (a producer
// and a consumer) whenever the consumer takes one or more of the producer's
// bundle results directly as inputs.  The match is strict SSA equality: the
// upstream `bundle(unbundle(X)...) -> X` canonicalization is what makes this
// shape appear in the first place.
//
// When the consumer fans in from more than one parallel producer (e.g. a
// per-device residual `mo.add` whose two operands come from two different
// upstream allreduce parallels), the pass picks the first producer (in
// consumer-input order) whose merge satisfies the dependence check.
// Outputs of the other parallel producers stay as ordinary non-shared
// inputs of the merged op and may be folded in by a later iteration of
// the pass driver.
//
// Mechanics:
//
//   * Inputs shared between consumer and producer short-circuit: the
//     consumer's block argument for that input is remapped to the
//     producer's corresponding yield operand, avoiding the bundle
//     round-trip.
//   * Non-shared consumer inputs are appended to the merged input list with
//     a fresh block arg each.
//   * The producer's body ops are cloned into the merged block, then the
//     consumer's body ops are cloned with consumer-side block args
//     rewritten per the remap above.  A fresh `mo.yield` joins the
//     surviving producer yields with the consumer's yields.
//
// Dead-yield elimination: any producer bundle result whose sole external
// user was the consumer's now-absorbed input is dropped from the merged
// yield and result list.  Without this the downstream MOGG epilogue fusion
// stalls on a spurious "live" output path out of the merged body.
//
// Buffer + chain policy (strict, matching what allreduce signal buffers
// actually need):
//
//   * Both sides have buffers + chain: buffer lists must be structurally
//     identical and the consumer's in-chain must be the producer's
//     out-chain with no other external users.  The merged op reuses the
//     producer's buffers and in-chain; its out-chain replaces both the
//     producer's and the consumer's.
//   * Only one side has buffers + chain: the merged op inherits that side.
//   * Neither: the merged op is chainless.
//   * Buffer mismatch: leave the pair unmerged.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "merge-adjacent-parallels"

namespace M::MO {

#define GEN_PASS_DEF_MERGEADJACENTPARALLELS
#include "GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

using namespace M;
using namespace M::MO;

namespace {

//===----------------------------------------------------------------------===//
// Small utilities
//===----------------------------------------------------------------------===//

/// Returns the number of bundle outputs (excluding the trailing chain, if
/// present).
static size_t numBundleResults(ParallelOp op) {
  return op.getNumResults() - (op.hasChain() ? 1 : 0);
}

/// Returns true if \p producer's buffer operand list is structurally
/// identical to \p consumer's (same SSA values, same order, same length).
static bool buffersMatch(ParallelOp producer, ParallelOp consumer) {
  auto producerBuffers = producer.getBuffers();
  auto consumerBuffers = consumer.getBuffers();
  if (producerBuffers.size() != consumerBuffers.size())
    return false;
  for (auto [producerBuffer, consumerBuffer] :
       llvm::zip(producerBuffers, consumerBuffers))
    if (producerBuffer != consumerBuffer)
      return false;
  return true;
}

/// Return true if \p root transitively depends on a result of \p target,
/// by performing a depth-first search of the operand closure of \p root.
static bool dependsOn(Value root, Operation *target) {
  llvm::DenseSet<Value> visited;
  llvm::SmallVector<Value> stack;
  stack.push_back(root);

  while (!stack.empty()) {
    Value value = stack.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    Operation *def = value.getDefiningOp();
    if (!def)
      continue;
    if (def == target)
      return true;
    llvm::append_range(stack, def->getOperands());
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Candidate discovery
//===----------------------------------------------------------------------===//

/// A single merge site.  `sharedInputs[i] = producerIdx` means the
/// consumer's input #i is the producer's result #producerIdx; nullopt means
/// the consumer's input #i is sourced elsewhere and must be carried into
/// the merged op.
struct MergeCandidate {
  ParallelOp producer;
  ParallelOp consumer;
  SmallVector<std::optional<size_t>> sharedInputs;
};

/// Returns true if any consumer input that is NOT a result of \p producer
/// transitively depends on \p producer's results, which would create a
/// dependence cycle if the consumer were merged into the producer.  This
/// is what disqualifies the wrong choice in the multi-producer case
/// (e.g. picking an upstream producer when a downstream producer's
/// output is also fed in through another consumer input).
static bool
nonSharedInputDependsOnProducer(ParallelOp consumer, ParallelOp producer,
                                ArrayRef<std::optional<size_t>> sharedInputs) {
  for (auto [i, input] : llvm::enumerate(consumer.getInputs())) {
    if (sharedInputs[i].has_value())
      continue;
    if (dependsOn(input, producer.getOperation()))
      return true;
  }
  return false;
}

/// If \p consumer can be merged into an upstream producer ParallelOp,
/// returns the candidate.  When the consumer fans in from multiple parallel
/// producers, iterates them in consumer-input order and returns the first
/// one that satisfies all checks; the pass driver's fixed-point loop then
/// picks up any further cascades.  Requires:
///
///   * At least one consumer input is a result of the chosen producer op.
///   * Producer and consumer share the same block.
///   * Every non-shared consumer input is independent of the chosen
///     producer's results (no dependence cycle through the producer).
///
/// Other parallel producers that the consumer fans in from are kept as
/// regular non-shared inputs of the merged op; they may merge in a later
/// iteration if their own dependence checks pass.
static std::optional<MergeCandidate> findMergeCandidate(ParallelOp consumer) {
  // Collect unique parallel producers in first-seen input order so the
  // selection is deterministic.
  SmallVector<ParallelOp> candidatesInOrder;
  llvm::SmallDenseSet<Operation *> seen;
  for (Value input : consumer.getInputs()) {
    auto prod = dyn_cast_or_null<ParallelOp>(input.getDefiningOp());
    if (!prod || prod.getOperation() == consumer.getOperation())
      continue;
    if (seen.insert(prod.getOperation()).second)
      candidatesInOrder.push_back(prod);
  }

  for (ParallelOp producer : candidatesInOrder) {
    if (producer->getBlock() != consumer->getBlock())
      continue;

    SmallVector<std::optional<size_t>> sharedInputs(
        consumer.getInputs().size());
    for (auto [i, input] : llvm::enumerate(consumer.getInputs()))
      if (input.getDefiningOp() == producer.getOperation())
        sharedInputs[i] = cast<OpResult>(input).getResultNumber();

    if (nonSharedInputDependsOnProducer(consumer, producer, sharedInputs))
      continue;

    return MergeCandidate{producer, consumer, std::move(sharedInputs)};
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Buffer + chain resolution
//===----------------------------------------------------------------------===//

struct BufferChainPlan {
  /// Ordered buffer operand list for the merged op.
  SmallVector<Value> buffers;
  /// In-chain for the merged op (empty = chainless).
  Value inChain;
};

/// Applies the strict buffer+chain rules described at the top of the file.
/// Returns std::nullopt if the candidate must be rejected.
/// NOTE: Once GEX-3527 lands, we no longer need to handle mo.buffer within
/// mo.parallel.
static std::optional<BufferChainPlan>
resolveBufferChain(const MergeCandidate &candidate) {
  ParallelOp producer = candidate.producer;
  ParallelOp consumer = candidate.consumer;
  BufferChainPlan plan;

  if (producer.hasChain() && consumer.hasChain()) {
    if (!buffersMatch(producer, consumer))
      return std::nullopt;
    if (consumer.getInChain() != producer.getOutChain())
      return std::nullopt;
    if (!producer.getOutChain().hasOneUse())
      return std::nullopt;
    plan.buffers.assign(producer.getBuffers().begin(),
                        producer.getBuffers().end());
    plan.inChain = producer.getInChain();
    return plan;
  }

  if (producer.hasChain()) {
    plan.buffers.assign(producer.getBuffers().begin(),
                        producer.getBuffers().end());
    plan.inChain = producer.getInChain();
    return plan;
  }
  if (consumer.hasChain()) {
    plan.buffers.assign(consumer.getBuffers().begin(),
                        consumer.getBuffers().end());
    plan.inChain = consumer.getInChain();
    return plan;
  }

  return plan; // neither has a chain
}

//===----------------------------------------------------------------------===//
// Merge plan
//===----------------------------------------------------------------------===//

/// All the bookkeeping needed to build the merged op and rewrite uses
/// afterwards.  Computed up front so the actual construction is
/// straightforward.
struct MergePlan {
  /// Merged input SSA values (producer's inputs + consumer's non-shared
  /// inputs).
  SmallVector<Value> mergedInputs;
  /// For each shared consumer input, the producer yield operand to
  /// substitute (for the non-shared entries, this stays null and
  /// `consumerAppendedArgIdx[bi]` is used instead).
  SmallVector<Value> consumerSharedYieldOperand;
  /// For each non-shared consumer input, the block-arg index in the merged
  /// block that carries that input's per-launch value.  Meaningless for
  /// shared inputs.
  SmallVector<size_t> consumerAppendedArgIdx;
  /// dropProducerYield[i] = true means the producer's bundle result #i has
  /// no external users beyond the now-absorbed consumer and can be omitted
  /// from the merged yield list.
  llvm::BitVector dropProducerYield;
  /// Merged result types (kept producer bundles, then consumer bundles,
  /// then optional trailing chain).
  SmallVector<Type> mergedResultTypes;
  /// For each producer bundle index, the merged result index it maps to,
  /// or nullopt if dropped.
  SmallVector<std::optional<size_t>> producerMergedIdx;
  /// Index of the first merged result corresponding to a consumer bundle
  /// result.
  size_t consumerMergedStart = 0;
};

/// Builds the merge plan.  The heavy lifting lives here so the actual IR
/// manipulation in `mergeCandidate` reads as a linear recipe.
static MergePlan planMerge(const MergeCandidate &candidate,
                           const BufferChainPlan &bufferChain) {
  ParallelOp producer = candidate.producer;
  ParallelOp consumer = candidate.consumer;
  Block &producerBody = producer.getBodyRegion().front();
  auto producerYield = cast<YieldOp>(producerBody.getTerminator());
  size_t producerNumBundles = numBundleResults(producer);
  size_t consumerNumBundles = numBundleResults(consumer);

  MergePlan plan;
  plan.mergedInputs.assign(producer.getInputs().begin(),
                           producer.getInputs().end());

  // Drop any yield in the producer that is only used by the consumer.
  plan.dropProducerYield.resize(producerNumBundles);
  for (size_t i = 0; i < producerNumBundles; ++i) {
    bool hasExternalUser = false;
    for (Operation *user : producer.getResult(i).getUsers())
      if (user != consumer.getOperation()) {
        hasExternalUser = true;
        break;
      }
    plan.dropProducerYield[i] = !hasExternalUser;
  }

  // For each consumer input, precompute either the producer yield operand
  // to substitute (shared case) or the merged-block-arg index it will
  // occupy (non-shared case).  This keeps `spliceBodies` a straight O(n)
  // rewrite.
  plan.consumerSharedYieldOperand.assign(consumer.getInputs().size(), Value{});
  plan.consumerAppendedArgIdx.assign(consumer.getInputs().size(), 0);
  size_t appendedIdx = producerBody.getNumArguments();
  for (auto [bi, input] : llvm::enumerate(consumer.getInputs())) {
    if (candidate.sharedInputs[bi].has_value()) {
      plan.consumerSharedYieldOperand[bi] =
          producerYield.getOperand(*candidate.sharedInputs[bi]);
      continue;
    }
    plan.mergedInputs.push_back(input);
    plan.consumerAppendedArgIdx[bi] = appendedIdx++;
  }

  // --- Merged result types + producer result index mapping ---
  plan.producerMergedIdx.assign(producerNumBundles, std::nullopt);
  for (size_t i = 0; i < producerNumBundles; ++i) {
    if (plan.dropProducerYield[i])
      continue;
    plan.producerMergedIdx[i] = plan.mergedResultTypes.size();
    plan.mergedResultTypes.push_back(producer.getResult(i).getType());
  }
  plan.consumerMergedStart = plan.mergedResultTypes.size();
  for (size_t i = 0; i < consumerNumBundles; ++i)
    plan.mergedResultTypes.push_back(consumer.getResult(i).getType());
  if (bufferChain.inChain)
    plan.mergedResultTypes.push_back(ChainType::get(producer.getContext()));

  return plan;
}

//===----------------------------------------------------------------------===//
// Merge
//===----------------------------------------------------------------------===//

/// Clones the producer's body ops then the consumer's body ops into
/// \p merged's block, using an IRMapping that wires the producer's block
/// args to the merged block args and the consumer's block args either to
/// the matching producer yield operand (for shared inputs, post-clone so
/// the cloned yield operand is used) or to the appended merged block arg
/// (for non-shared inputs).  Emits the combined yield.
static void spliceBodies(ParallelOp producer, ParallelOp consumer,
                         ParallelOp merged, const MergeCandidate &candidate,
                         const MergePlan &plan) {
  Block &producerBody = producer.getBodyRegion().front();
  Block &consumerBody = consumer.getBodyRegion().front();
  Block *mergedBlock = &merged.getBodyRegion().front();
  auto producerYield = cast<YieldOp>(producerBody.getTerminator());
  auto consumerYield = cast<YieldOp>(consumerBody.getTerminator());

  // Clone the producer's body into the merged parallel op.
  IRMapping mapping;
  for (auto [oldArg, newArg] :
       llvm::zip(producerBody.getArguments(), mergedBlock->getArguments()))
    mapping.map(oldArg, newArg);

  OpBuilder bodyBuilder(merged.getContext());
  bodyBuilder.setInsertionPointToEnd(mergedBlock);
  for (Operation &op : producerBody.without_terminator())
    bodyBuilder.clone(op, mapping);

  // With the producer's body cloned, the consumer's shared-input block
  // args map to the *cloned* producer yield operand (via
  // `mapping.lookupOrDefault`).  Non-shared consumer args map to merged
  // block args at positions precomputed in the plan.
  for (auto [bi, consumerArg] : llvm::enumerate(consumerBody.getArguments())) {
    if (candidate.sharedInputs[bi].has_value())
      mapping.map(consumerArg,
                  mapping.lookupOrDefault(plan.consumerSharedYieldOperand[bi]));
    else
      mapping.map(consumerArg,
                  mergedBlock->getArgument(plan.consumerAppendedArgIdx[bi]));
  }
  for (Operation &op : consumerBody.without_terminator())
    bodyBuilder.clone(op, mapping);

  SmallVector<Value> yieldValues;
  for (auto [i, v] : llvm::enumerate(producerYield.getOperands()))
    if (!plan.dropProducerYield[i])
      yieldValues.push_back(mapping.lookupOrDefault(v));
  for (Value v : consumerYield.getOperands())
    yieldValues.push_back(mapping.lookupOrDefault(v));
  YieldOp::create(bodyBuilder, merged.getLoc(), yieldValues);
}

/// Routes all uses of \p producer's and \p consumer's results to the
/// corresponding merged results, then erases both originals.
static void replaceAndErase(ParallelOp producer, ParallelOp consumer,
                            ParallelOp merged, const MergePlan &plan,
                            const BufferChainPlan &bufferChain) {
  size_t producerNumBundles = numBundleResults(producer);
  size_t consumerNumBundles = numBundleResults(consumer);

  for (size_t i = 0; i < producerNumBundles; ++i)
    if (plan.producerMergedIdx[i])
      producer.getResult(i).replaceAllUsesWith(
          merged.getResult(*plan.producerMergedIdx[i]));
  for (size_t i = 0; i < consumerNumBundles; ++i)
    consumer.getResult(i).replaceAllUsesWith(
        merged.getResult(plan.consumerMergedStart + i));
  if (bufferChain.inChain) {
    Value mergedOutChain = merged.getResult(merged.getNumResults() - 1);
    if (producer.hasChain())
      producer.getOutChain().replaceAllUsesWith(mergedOutChain);
    if (consumer.hasChain())
      consumer.getOutChain().replaceAllUsesWith(mergedOutChain);
  }

  consumer.erase();
  producer.erase();
}

/// Merges \p candidate.consumer into \p candidate.producer.
static void mergeCandidate(MergeCandidate &candidate,
                           const BufferChainPlan &bufferChain,
                           OpBuilder &builder) {
  MergePlan plan = planMerge(candidate, bufferChain);

  builder.setInsertionPoint(candidate.producer);
  ParallelOp merged = ParallelOp::create(
      builder, candidate.producer.getLoc(), plan.mergedInputs,
      bufferChain.buffers, bufferChain.inChain, plan.mergedResultTypes);
  spliceBodies(candidate.producer, candidate.consumer, merged, candidate, plan);
  replaceAndErase(candidate.producer, candidate.consumer, merged, plan,
                  bufferChain);
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

/// Walks every ParallelOp and tries to merge it into its producer, if any.
/// Returns true if at least one merge fired.
static bool runOneIteration(Operation *root, OpBuilder &builder) {
  SmallVector<ParallelOp> parallels;
  root->walk([&](ParallelOp op) { parallels.push_back(op); });

  bool changed = false;
  for (ParallelOp parallelOp : parallels) {
    // May have been erased by an earlier merge this iteration.
    if (!parallelOp.getOperation()->getBlock())
      continue;

    auto candidate = findMergeCandidate(parallelOp);
    if (!candidate)
      continue;

    auto bufferChain = resolveBufferChain(*candidate);
    if (!bufferChain)
      continue;

    mergeCandidate(*candidate, *bufferChain, builder);
    changed = true;
  }
  return changed;
}

struct MergeAdjacentParallelsPass
    : public impl::MergeAdjacentParallelsBase<MergeAdjacentParallelsPass> {
  using MergeAdjacentParallelsBase::MergeAdjacentParallelsBase;

  void runOnOperation() override {
    OpBuilder builder{&getContext()};
    while (runOneIteration(getOperation(), builder)) {
    }
  }
};

} // namespace
