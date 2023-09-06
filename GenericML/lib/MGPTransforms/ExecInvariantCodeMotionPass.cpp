//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/MGPTransforms.h"

#include "Cache/CacheDialect/CacheAttrs.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPUtils.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "GraphRT/MEFDialect/MEFOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "./Utils.h"

using namespace mlir;

namespace M::MGP {

// Generate support code for ExecInvariantCodeMotionPass.
#define GEN_PASS_DEF_EXECINVARIANTCODEMOTIONPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// ExecInvariantCodeMotionPass
//===----------------------------------------------------------------------===//

/// This pass is very similar to loop invariant code motion (if one thinks of
/// the init phase as pre-loop code, and the exec phase as the loop body).
///
/// However, there is no shared scope between the init exec phases, and this
/// causes some critical differences in the pass implementation (otherwise we
/// won't be able to lift anything useful at all):
///
/// 1. We want to lift buffer-producing ops into the init phase. In order to
/// reference such lifted buffers in the exec phase, we need to leverage the
/// buffer cache mechanism, i.e., cache in the init phase, and look up the cache
/// in the exec phase. Note that
///   (a). if all users of a lifted buffer are also lifted, we don't cache such
///   buffers, thereby minimizing the number of buffer cache ops inserted.
///   (b). if no user of a buffer op is lifted, we don't lift the buffer op
///   because it merely adds more caching without saving us any computation.
///
/// 2. We want to lift ops that modifies the lifted buffers into the init phase
/// (e.g., matmul weights pre-packing). Usually such ops produces a chain; in
/// order to preserve control dependency, we must
///   a. update init phase to output a chain that accounts for the lifted chain.
///   b. update exec phase to use the input chain instead of the lifted chain.
///
/// 3. We want to lift ops that produces non-buffer output, in order to
/// reference such lifted values in the exec phase, we need to perform
/// re-computation, i.e., leave a clone of the computation inside exec phase,
/// and lift the original computation (unless we introduce caching mechanism for
/// these output types too). For instance, output spec of matmul weight packing
/// is used by both the lifted packing computation, and the actual matmul
/// computation in the exec phase.
class ExecInvariantCodeMotionPass
    : public impl::ExecInvariantCodeMotionPassBase<
          ExecInvariantCodeMotionPass> {
  using ExecInvariantCodeMotionPassBase::ExecInvariantCodeMotionPassBase;

  static void expandByTransitiveInputs(DenseSet<Operation *> &ops) {
    SmallVector<Operation *> worklist(ops.begin(), ops.end());
    while (!worklist.empty()) {
      Operation *currOp = worklist.pop_back_val();
      ops.insert(currOp);
      for (Value input : currOp->getOperands()) {
        if (Operation *inputOp = input.getDefiningOp()) {
          if (auto [_, inserted] = ops.insert(inputOp); inserted)
            worklist.push_back(inputOp);
        }
      }
    }
  }

  // NOTE these predicates are intentionally restricted. We will gradually
  // relax and maybe generalize them later.

  /// Predicate for liftable buffer op, i.e., operation which produces a single
  /// buffer, which can be lifted, cached, and used again in the exec block.
  static bool isLiftableBufferOp(Operation *op) {
    return isLiftableTrivialBufferOp(op) || isa<CpuBufferAllocSpecOp>(op);
  }

  // Predicate for liftable buffer op that are expected to have trivial cost (so
  // that lifting them won't save any computation if all of their uses are
  // unliftable)
  static bool isLiftableTrivialBufferOp(Operation *op) {
    return isa<CpuBufferAllocStaticOp, CpuBufferConstantOp>(op);
  }

  /// Predicate for liftable chain op, i.e., operation which produces a single
  /// chain type, which can be lifted, merged into the final output chain of
  /// init block, so that dependency is preserved in the exec block.
  static bool isLiftableChainOp(Operation *op) {
    return op->getNumResults() == 1 &&
           isa<MEF::ChainType>(op->getResultTypes().front());
  }

  /// Predicate for re-computable op, i.e., operation whose results cannpt be
  /// cached, but we'd like to lift it by re-computing in the exec block,
  /// because it could enable more lifting.
  static bool isRecomputableOp(Operation *op) {
    // TODO find an alternative to hard-coding this; it might require some
    // cost-model.
    static llvm::SmallDenseSet<StringRef> recomputableOps{
        CpuTensorSpecCreateOp::getOperationName(),
    };
    // We allow compiled kernels to be recomputed if they return indices,
    // which signals they are just shape functions.
    if (auto executeOp = dyn_cast<CpuKernelExecuteOp>(op)) {
      if (!executeOp.getOutDimensions().empty())
        return true;
    }
    return recomputableOps.contains(op->getName().getStringRef());
  }

  /// Predicate for checking potential side-effect of an op. The term
  /// "side-effect" is used a bit loosely in our context, e.g., a buffer
  /// allocation technically has side-effect, but we consider it as side-effect
  /// free for lifting purposes. TODO consider using a more accurate term.
  static bool doesOpHaveSideEffect(Operation *op) {
    // TODO(#12221) remove hard-coded set once `mop` ops are registered.
    // TODO(#12270) find a better model for the properties to inspect here
    static llvm::SmallDenseSet<StringRef> unregisteredPureOps{
        CpuKernelExecuteOp::getOperationName(),
        CpuBufferAllocSpecOp::getOperationName(),
        CpuBufferAllocStaticOp::getOperationName(),
    };
    return !isPure(op) &&
           !unregisteredPureOps.contains(op->getName().getStringRef());
  }

  /// Initializes the following fields
  /// - `liftableBufferOpToLiftableUseCount`
  void initLiftableUseCountMaps(ExecuteState &executeState) {
    for (Operation &op : executeState.execute->getOperations()) {
      if (isLiftableBufferOp(&op)) {
        liftableBufferOpToLiftableUseCount[&op] =
            llvm::range_size(op.getUses());
      }
    }
  }

  /// Push `op` onto `worklist` if it hasn't been tracked by `unliftableOps`.
  void pushOntoWorklist(Operation *op) {
    if (auto [_, inserted] = unliftableOps.insert(op); inserted)
      worklist.push_back(op);
  }

  void processOpFromWorklist() {
    Operation *currOp = worklist.pop_back_val();

    // Since `currOp cannot be lifted, its users definitely cannot be lifted
    for (Operation *user : currOp->getUsers())
      pushOntoWorklist(user);

    // Normally we must lift the inputs of `currOp` as well, but there are a
    // few exceptions
    for (Value input : currOp->getOperands()) {
      // Skip if input doesn't come from an op, or it was already processed
      Operation *inputOp = input.getDefiningOp();
      if (!inputOp || unliftableOps.contains(inputOp))
        continue;

      // If input came from an op that produced a single buffer, we allow
      // certain such inputs to be lifted (remaining uses will be updated by
      // leveraging the buffer cache mechanism)
      if (isLiftableBufferOp(inputOp)) {
        // For trivial buffers ops, don't lift them (mark as unliftable) if
        // _all_ of their uses are unliftable, because we don't save any
        // computation by lifting such buffer ops.
        int64_t count = --liftableBufferOpToLiftableUseCount[inputOp];
        assert(count >= 0 && "underflow in liftable buffer op use count");
        if (count == 0 && isLiftableTrivialBufferOp(inputOp))
          pushOntoWorklist(inputOp);
        continue;
      }

      // If input came from an op that produced a single chain, we allow
      // certain such inputs to be lifted; the lifted chains will be merged at
      // the end of init phase, and remaining uses in the exec phase will use
      // the input chain instead, thereby preserving dependencies.
      if (isLiftableChainOp(inputOp))
        continue;

      // Although this input must stay inside the exec phase, we could still
      // lift certain such inputs by cloning/re-computing them in the exec
      // phase, and updating uses inside exec phase to the cloned one.
      if (isRecomputableOp(inputOp)) {
        opsToRecompute.insert(inputOp);
        continue;
      }

      // This op must stay in exec as is, no matter what its other users say.
      pushOntoWorklist(inputOp);
    }
  }

  /// Initializes the following fields
  /// - `unliftableOps`
  /// - `recomputableOps` (partially, only the roots of recomputations)
  void findUnliftableOpsAndOpsToRecompute(ExecuteState &executeState) {
    // We can't lift any op that uses the extra arguments from execute phase.
    for (BlockArgument extraArg : executeState.getExtraArgs()) {
      for (Operation *user : extraArg.getUsers())
        pushOntoWorklist(user);
    }
    // We can't lift any op that has side effect (e.g., debug prints)
    for (Operation &op : executeState.execute->getOperations())
      if (doesOpHaveSideEffect(&op))
        pushOntoWorklist(&op);
    // We can't lift the terminator in execute phase.
    pushOntoWorklist(executeState.execute->getTerminator());

    while (!worklist.empty())
      processOpFromWorklist();
  }

  void cloneOpsToRecompute() {
    // Filter `opsToRecompute` because some of them might have become
    // unliftable.
    llvm::set_subtract(opsToRecompute, unliftableOps);

    // Find all the ops that need to be recomputed, because currently
    // `opsToRecompute` might only contain the root of such computations.
    expandByTransitiveInputs(opsToRecompute);

    // Clone the re-computed ops and wire up the uses.
    IRMapping irMap;
    OpBuilder execBuilder(&getContext());
    for (Operation *op : opsToRecompute) {
      // Although not required, this makes debugging easier.
      execBuilder.setInsertionPoint(op);
      Operation *clonedOp = execBuilder.clone(*op, irMap);
      // Replace old uses that will remain in exec block
      op->replaceUsesWithIf(clonedOp, [&](OpOperand &operand) {
        return unliftableOps.contains(operand.getOwner());
      });
      // We don't want to lift the clonedOp, which is meant to stay in the exec
      // phase and preserving validity of its users there.
      unliftableOps.insert(clonedOp);
    }
  }

  /// Move the ops that need to be lifted from exec to init phase, with some
  /// post processing on chains to preserve control dependencies.
  void moveLiftableOps(InitState &initState, ExecuteState &executeState) {
    auto initOutputOp = cast<MEF::OutputOp>(*initState.init->rbegin());
    Value initOriginalChain = initState.getFinalChain();
    Value execInputChain = executeState.getRootChain();
    Value execInputCtx = executeState.getContext();
    OpBuilder initBuilder(initOutputOp);
    OpBuilder execBuilder(&getContext());
    Location outputLoc = initOutputOp.getLoc();
    // The proper input chain is produced at the end, so we use a dummy as we
    // construct the buffer cache ops, and replace it at the end.
    auto dummyBufferCacheInputChain =
        initBuilder.create<GRT::ChainCreateOp>(outputLoc);
    // Note that dependencies among lifted ops remain as buffer (e.g., any
    // buffer writing op still refers to the buffer directly, instead of going
    // through the cache like remaining uses in the exec block).
    //
    // However, we must be careful with control dependencies because
    // 1. buffer caching does a deep copy, requiring actions on buffer to
    //    complete first (thus the `potentialBufferActionChains` list).
    // 2. we must collect the new output dependencies introduced to the init
    //    phase (thus the `finalOutChains` list)
    // TODO(#12041) this may be too aggressive, relax via some analysis
    SmallVector<Value> potentialBufferActionChains;
    SmallVector<Value> finalOutChains{initOriginalChain};

    // A local helper; we can't make this a member method and pass it as lambda.
    auto isOpOperandUnliftable = [this](OpOperand &operand) {
      return unliftableOps.contains(operand.getOwner());
    };

    // We need `make_early_inc_range` because we might move `op`.
    for (Operation &op :
         llvm::make_early_inc_range(executeState.execute->getOperations())) {
      if (unliftableOps.contains(&op))
        continue;

      // Although not required, this makes debugging easier.
      execBuilder.setInsertionPoint(&op);

      if (isLiftableChainOp(&op)) {
        // Replace old uses that will remain in exec block
        op.getResult(0).replaceUsesWithIf(execInputChain,
                                          isOpOperandUnliftable);

      } else if (isLiftableBufferOp(&op) &&
                 llvm::any_of(op.getUses(), isOpOperandUnliftable)) {
        // Collect output chains of any lifted user that writes to this buffer
        for (OpOperand &use : op.getUses())
          if (!isOpOperandUnliftable(use) && isBufferWritingOpOperand(use))
            for (Value res : use.getOwner()->getResults())
              if (isa<MEF::ChainType>(res.getType()))
                potentialBufferActionChains.push_back(res);

        // If this liftable buffer op has unliftable users, insert
        // - `cpu.buffer.cache` in the init phase
        // - `cpu.buffer.get_cache` in the exec phase
        size_t currBufferSlot = initState.allocSlot();
        auto bufferCacheOutChain = initBuilder.create<CpuBufferCacheOp>(
            op.getLoc(), dummyBufferCacheInputChain, initState.getContext(),
            op.getResult(0), currBufferSlot);
        auto bufferType = cast<CpuBufferType>(op.getResultTypes().front());
        auto bufferGetCacheOp = execBuilder.create<CpuBufferGetCachedOp>(
            op.getLoc(), execInputChain, execInputCtx, currBufferSlot,
            bufferType.getRawSize().getUInt());

        // Keep track of this as final chains because buffer caching will happen
        // at the end of init phase.
        finalOutChains.push_back(bufferCacheOutChain);

        // Replace old uses that will remain in exec block
        op.replaceUsesWithIf(bufferGetCacheOp, isOpOperandUnliftable);
      }

      // Finally, lift this op into the init phase.
      op.moveBefore(initOutputOp);
    }

    // -------------------------------------------------------------
    // post-processings
    // -------------------------------------------------------------
    // Replace all uses of exec input chain and context for the lifted ops
    execInputChain.replaceUsesWithIf(
        initOriginalChain, [&](OpOperand &operand) {
          return operand.getOwner()->getBlock() == initState.init;
        });
    execInputCtx.replaceUsesWithIf(
        initState.contextCreateOp, [&](OpOperand &operand) {
          return operand.getOwner()->getBlock() == initState.init;
        });

    // Basically produces the following
    //   %merged_action_ch = grt.chain.create %action_ch0, %action_ch1, ...
    //   %final_ch = grt.chain.create %original_ch, %cache_out_ch0,
    //   %cache_out_ch1, ... mef.output %final_ch, ...
    auto mergedBufferActionChain = initBuilder.create<GRT::ChainCreateOp>(
        outputLoc, potentialBufferActionChains);
    auto mergedFinalOutChain =
        initBuilder.create<GRT::ChainCreateOp>(outputLoc, finalOutChains);
    initState.updateFinalChain(mergedFinalOutChain);

    // And update the dummy chain
    dummyBufferCacheInputChain.replaceAllUsesWith(
        mergedBufferActionChain.getResult());
  }

  void runOnOperation() override {
    // Clear state because the instance might be reused
    unliftableOps.clear();
    worklist.clear();
    opsToRecompute.clear();
    liftableBufferOpToLiftableUseCount.clear();

    ModelOp model = getOperation();
    InitState initState(model.getContext(), model.getInitBlock());
    ExecuteState executeState(model.getContext(), model.getExecuteBlock());
    initLiftableUseCountMaps(executeState);
    findUnliftableOpsAndOpsToRecompute(executeState);
    cloneOpsToRecompute();
    moveLiftableOps(initState, executeState);
  }

  // Worklist contains ops whose inputs/users could potentially become
  // unliftable, because these ops themselves are deemed unliftable.
  SmallVector<Operation *> worklist;
  // This contains all the ops we ever push onto `worklist`, so we only process
  // each op once, have a way to query for liftability later.
  DenseSet<Operation *> unliftableOps;
  // This contains ops we must re-compute (i.e., compute in both init and exec
  // phases) so we can enable more lifting. TODO(#16768)
  DenseSet<Operation *> opsToRecompute;
  // Maps each liftable buffer op in the exec block to the # of liftable uses
  // (not users). Start by assuming all users are liftable, and decrement as we
  // find unliftable users.
  DenseMap<Operation *, int64_t> liftableBufferOpToLiftableUseCount;
};

} // namespace
} // namespace M::MGP
