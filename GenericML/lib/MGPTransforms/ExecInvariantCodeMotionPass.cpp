//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/MGPTransforms.h"

#include "GenericML/MGPDialect/BufferBuilder.h"
#include "GenericML/MGPDialect/MGPDialect.h"
#include "GenericML/MGPDialect/MGPOpInterfaces.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTensorInterface.h"
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
#include "llvm/Support/Debug.h"

#include "./Utils.h"

using namespace mlir;

#define DEBUG_TYPE "exec-invariant-code-motion"

namespace M::MGP {

// Generate support code for ExecInvariantCodeMotionPass.
#define GEN_PASS_DEF_EXECINVARIANTCODEMOTIONPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// ExecInvariantCodeMotionPass
//===----------------------------------------------------------------------===//

class ExecInvariantCodeMotionPass
    : public impl::ExecInvariantCodeMotionPassBase<
          ExecInvariantCodeMotionPass> {
  using ExecInvariantCodeMotionPassBase::ExecInvariantCodeMotionPassBase;

  //===----------------------------------------------------------------------===//
  // Predicates
  //===----------------------------------------------------------------------===//

  // NOTE these predicates are intentionally restricted. We will gradually
  // relax and maybe generalize them later.

  /// Returns true if op returns a buffer and could be moved to the 'init'
  /// block.
  static bool isLiftableBufferOp(Operation *op) {
    return isLiftableTrivialBufferOp(op) || isa<CpuBufferAllocSpecOp>(op);
  }

  /// Returns true if op returns a buffer and could be moved to the 'init'
  /// block, however only if some other lifted op in the 'init' block needs
  /// the buffer. Otherwise, assume the cost of caching and retrieving the
  /// buffer outweighs the cost of simply executing the op.
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

  /// Returns true if the result of op cannot be cached, but it is ok for the
  /// op to be executed in both the 'init' and 'execute' blocks. Doing so will
  /// allow downstream ops to also be lifted to the 'init' block.
  static bool isRecomputableOp(Operation *op) {
    // We allow compiled kernels to be recomputed if they return indices,
    // which signals they are just shape functions. Note that only CPU
    // kernels can do this, hence the hard-coded op type.
    if (auto executeOp = dyn_cast<CpuKernelExecuteOp>(op)) {
      if (!executeOp.getOutDimensions().empty())
        return true;
    }
    // It is ok to recompute tensor specs.
    if (isa<CpuTensorSpecCreateOp>(op))
      return true;

    return false;
  }

  /// Returns true if the execution of op can be safely reordered in the
  /// model. This could be because the op is side-effect free, or any side
  /// effects can only be observed cumulatively at the end of the model
  /// (eg buffer allocation).
  static bool isReorderableOp(Operation *op) {
    // TODO(#12221) remove hard-coded set once `mop` ops are registered.
    // TODO(#12270) find a better model for the properties to inspect here

    if (isPure(op))
      return true;

    if (isLiftableBufferOp(op))
      return true;

    // We'll consider kernel launches to be re-orderable, though technically
    // they may have side-effects.
    if (isa<MGP::CpuKernelExecuteOp>(op))
      return true;

    // No luck.
    return false;
  }

  //===----------------------------------------------------------------------===//
  // Utils
  //===----------------------------------------------------------------------===//

  static void expandByTransitiveInputs(DenseSet<Operation *> &ops) {
    SmallVector<Operation *> worklist(ops.begin(), ops.end());
    while (!worklist.empty()) {
      Operation *currOp = worklist.pop_back_val();

      // Recompute only if there are no other ways of hoisting the
      // operation.
      if (!isLiftableBufferOp(currOp) && !isLiftableChainOp(currOp))
        ops.insert(currOp);

      for (Value input : currOp->getOperands()) {
        if (Operation *inputOp = input.getDefiningOp()) {
          // Recompute only if there are no other ways of hoisting the
          // operation.
          if (inputOp && !isLiftableBufferOp(inputOp) &&
              !isLiftableChainOp(inputOp)) {
            ops.insert(inputOp);
          }
          if (inputOp)
            worklist.push_back(inputOp);
        }
      }
    }
  }

  //===----------------------------------------------------------------------===//
  // Processing
  //===----------------------------------------------------------------------===//

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
    for (Operation &op : executeState.execute->getOperations()) {
      if (!isReorderableOp(&op))
        pushOntoWorklist(&op);
    }
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
    OpBuilder initBuilder(initOutputOp);
    OpBuilder execBuilder(&getContext());
    Location outputLoc = initOutputOp.getLoc();

    // This placeholder chain is used to gate storing all the lifted buffers
    // into the cache until all the side effects on those buffers have
    // completed. We'll replace this placeholder with the actual chain once
    // we've finished lifting all the ops.
    auto placeholderBufferCacheInputChain =
        initBuilder.create<GRT::ChainCreateOp>(outputLoc);

    // TODO(#12041) The following chain dependencies may be too aggressive,
    // relax via some analysis

    // Collects additional chains (converted to !mef.chain if required)
    // which are ready when all side effects for lifted buffers have completed.
    // We'll need to gate adding buffers to the !mgp.context cache slots until
    // ALL these side effects have completed.
    SmallVector<Value> potentialBufferActionChains;

    // Collects additional chain (always as !mef.chains) for inserting lifted
    // buffers into the !mgp.context cache. These will need to be joined into
    // the final 'init' block output chain.
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

      BufferBuilder *builder =
          TensorBufferDialectInterface::getBuilderForOp(*builderContext, &op);

      if (isLiftableChainOp(&op)) {
        Value oldChain = op.getResult(0);
        // Since we are moving op to the 'init' block, all references to its
        // oldChain must be replaced with (the device appropriate form of) the
        // 'execute' block's root chain. We'll let CSE clean up any duplication.
        Value newChain = builder->fromMEFChain(
            executeState.builder, op.getLoc(), executeState.getRootChain());
        oldChain.replaceUsesWithIf(newChain, isOpOperandUnliftable);

      } else if (isLiftableBufferOp(&op) &&
                 llvm::any_of(op.getUses(), isOpOperandUnliftable)) {
        // Collect the output chains (in !mef.chain form) for all the ops
        // which may write to the buffer. We can't safely cache the buffer
        // in the 'init' block until all those ops have completed.
        // However, rather than tracking this for each buffer individually,
        // we instead simply wait for *all* such ops to have completed in
        // the 'init' block before *all* buffers are cached.
        for (OpOperand &use : op.getUses()) {
          // TODO: This appears to be unsound: it is possible for one of the
          // writing uses of buffer to be unliftable, in which case it will
          // be rewritten to write to the buffer retrieved from the cache.
          // But cached buffers should never be mutated.
          if (!isOpOperandUnliftable(use) && isBufferWritingOpOperand(use)) {
            for (Value res : use.getOwner()->getResults()) {
              if (isa<MEF::MEFChainTypeInterface>(res.getType())) {
                potentialBufferActionChains.push_back(
                    builder->toMEFChain(initBuilder, op.getLoc(), res));
              }
            }
          }
        }

        // If this liftable buffer op has unliftable users:
        //  - cache the buffer in the 'init' block, but only once all the
        //    chains associated with side effects on all the buffers have
        //    completed.
        //  - recover the cached buffer in the 'execute' block.
        //  - replace all uses with the recovered buffer.
        auto origBuffer = op.getResult(0);
        auto [bufferCacheOutChain, currBufferSlot] = builder->cache(
            initBuilder, op.getLoc(), placeholderBufferCacheInputChain,
            initState.contextCreateOp, origBuffer);
        auto newBuffer = builder->getCached(
            execBuilder, op.getLoc(), executeState.getRootChain(),
            executeState.getContext(), currBufferSlot, origBuffer.getType());

        // Keep track of this as final chains because buffer caching will happen
        // at the end of init phase.
        finalOutChains.push_back(bufferCacheOutChain);

        // Replace old uses that will remain in exec block
        LLVM_DEBUG(llvm::dbgs() << "replace old buffer:\n"
                                << origBuffer << "\nwith:\n"
                                << newBuffer << "\nvia cached:\n"
                                << bufferCacheOutChain << "\n");
        origBuffer.replaceUsesWithIf(newBuffer, isOpOperandUnliftable);
      }

      // Finally, lift this op into the init phase.
      op.moveBefore(initOutputOp);
    }

    // -------------------------------------------------------------
    // Post-processing
    // -------------------------------------------------------------

    // Replace all uses of the 'execute' block's input chain with the output
    // chain of the 'init' block BEFORE we updated in above.
    executeState.getRootChain().replaceUsesWithIf(
        initOriginalChain, [&](OpOperand &operand) {
          return operand.getOwner()->getBlock() == initState.init;
        });

    // Similarly, replace all uses of the 'execute' block's input context
    // with the context from the 'init' block.
    executeState.getContext().replaceUsesWithIf(
        initState.contextCreateOp, [&](OpOperand &operand) {
          return operand.getOwner()->getBlock() == initState.init;
        });

    // Replace the caching placeholder chain with the !mef.chain which is
    // ready only when all side effects on all buffers moved to the 'init'
    // block have completed.
    placeholderBufferCacheInputChain.replaceAllUsesWith(
        initBuilder
            .create<GRT::ChainCreateOp>(outputLoc, potentialBufferActionChains)
            .getResult());

    // Make sure the final output !mef.chain waits for both the original
    // 'init' block output chain AND all the chains for the caching ops
    // we just inserted.
    auto mergedFinalOutChain =
        initBuilder.create<GRT::ChainCreateOp>(outputLoc, finalOutChains);
    initState.updateFinalChain(mergedFinalOutChain);
  }

  //===----------------------------------------------------------------------===//
  // Entry point
  //===----------------------------------------------------------------------===//

  void runOnOperation() override {
    assert(!builderContext && worklist.empty() && unliftableOps.empty() &&
           opsToRecompute.empty() &&
           liftableBufferOpToLiftableUseCount.empty());
    ModelOp modelOp = getOperation();
    InitState initState(modelOp.getContext(), modelOp.getInitBlock());
    ExecuteState executeState(modelOp.getContext(), modelOp.getExecuteBlock());

    // Recover type and buffer builders appropriate for the model.
    BufferBuilderContext theBuilderContext =
        TensorBufferDialectInterface::getAllBufferBuilders(modelOp);
    builderContext = &theBuilderContext;

    initLiftableUseCountMaps(executeState);
    findUnliftableOpsAndOpsToRecompute(executeState);
    cloneOpsToRecompute();
    moveLiftableOps(initState, executeState);

    // Clear state because the instance might be reused
    builderContext = nullptr;
    worklist.clear();
    unliftableOps.clear();
    opsToRecompute.clear();
    liftableBufferOpToLiftableUseCount.clear();
  }

  //===----------------------------------------------------------------------===//
  // State
  //===----------------------------------------------------------------------===//

  /// All the type and buffer builders we'll need while working on the model.
  BufferBuilderContext *builderContext = nullptr;
  /// Worklist contains ops whose inputs/users could potentially become
  /// unliftable, because these ops themselves are deemed unliftable.
  SmallVector<Operation *> worklist;
  /// This contains all the ops we ever push onto `worklist`, so we only process
  /// each op once, have a way to query for liftability later.
  DenseSet<Operation *> unliftableOps;
  /// This contains ops we must re-compute (i.e., compute in both init and exec
  /// phases) so we can enable more lifting. TODO(#16768)
  DenseSet<Operation *> opsToRecompute;
  /// Maps each liftable buffer op in the exec block to the # of liftable uses
  /// (not users). Start by assuming all users are liftable, and decrement as we
  /// find unliftable users.
  DenseMap<Operation *, int64_t> liftableBufferOpToLiftableUseCount;
};

} // namespace
} // namespace M::MGP
