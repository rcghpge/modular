//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "./Utils.h"

#include "GenericML/MGPDialect/MGPUtils.h"
#include "mlir/IR/Visitors.h"

using namespace M;
using namespace M::MGP;

/// Non-recursively visit inputs of op in dfs order, accumulating post-order
/// visits into dataflowOrder.
static void visitInDataflowOrder(Operation *op,
                                 SmallVectorImpl<Operation *> &dataflowOrder,
                                 DenseSet<Operation *> &visited) {
  if (!visited.insert(op).second)
    return;

  struct Frame {
    Operation *frameOp;
    size_t nextOperandIndex;
    Frame(Operation *frameOp, size_t nextOperandIndex)
        : frameOp(frameOp), nextOperandIndex(nextOperandIndex) {}
  };
  SmallVector<Frame> stack;

  // Start with first operand of op, if any.
  stack.emplace_back(op, 0);

  while (!stack.empty()) {
    Frame frame = stack.back();
    stack.pop_back();
    if (frame.nextOperandIndex >= frame.frameOp->getNumOperands()) {
      // We've visited all the inputs of frame.parent
      dataflowOrder.push_back(frame.frameOp);
    } else {
      // We're about to visit frame.parent->getOperand(frame.nextInputIndex),
      // so remember we need to continue with next operand when done, if any.
      stack.emplace_back(frame.frameOp, frame.nextOperandIndex + 1);

      Value operand = frame.frameOp->getOperand(frame.nextOperandIndex);
      if (Operation *inputOp = operand.getDefiningOp()) {
        if (visited.insert(inputOp).second)
          // Start with first operand of inputOp, if any.
          stack.emplace_back(inputOp, 0);
      }
    }
  }
}

SmallVector<Operation *> M::MGP::getDataflowOrder(Block *block) {
  SmallVector<Operation *> dataflowOrder;
  DenseSet<Operation *> visited;
  for (Operation &op : block->getOperations())
    visitInDataflowOrder(&op, dataflowOrder, visited);
  return dataflowOrder;
}

bool M::MGP::isAtMostOnce(Operation *op) {
  op = op->getParentOp();
  while (op) {
    if (auto modelOp = dyn_cast<ModelOp>(op))
      return true;
    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
      if (cfOp.getFlavor() == GRT::kLoopLike)
        return false;
    }
    op = op->getParentOp();
  };
  return true;
}

SmallVector<Value> M::MGP::getIndicatingResults(Operation *op) {
  if (!isa<MEF::NonStrictOp>(op))
    // All results are indicating for strict ops.
    return op->getResults();

  // Only chain results, if any, are indicating for non-strict ops.
  SmallVector<Value> results;
  for (Value result : op->getResults()) {
    if (isa<MEF::ChainType>(result.getType()))
      results.emplace_back(result);
  }
  return results;
}

Value M::MGP::getRepresentativeIndicatingResult(Operation *op) {
  SmallVector<Value> results = getIndicatingResults(op);
  if (results.empty())
    return {};
  for (Value result : results) {
    if (isa<MEF::ChainType>(result.getType()))
      return result;
  }
  return results.front();
}

SmallVector<Value> M::MGP::getGatingOperands(Operation *op) {
  SmallVector<Value> results;
  for (OpOperand &opOperand : op->getOpOperands()) {
    if (isGatingOperand(&opOperand))
      results.emplace_back(opOperand.get());
  }
  return results;
}

bool M::MGP::isGatingOperand(const OpOperand *opOperand) {
  Operation *op = opOperand->getOwner();

  if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
    // Only the extra strict operands are gating for control flow ops.
    auto extraOperands = cfOp.getStrictExtraOperands();
    return llvm::find(extraOperands, opOperand->get()) != extraOperands.end();
  }

  if (isa<MEF::OutputOp>(op))
    // Nothing gates a mef.output.
    return false;

  if (!isa<MEF::NonStrictOp>(op))
    // All operands are gating for strict ops.
    return true;

  // Only chain operands, if any, are indicating for non-strict ops.
  return isa<MEF::ChainType>(opOperand->get().getType());
}

SmallVector<OpOperand *> M::MGP::allUsingOpOperands(Value buffer,
                                                    bool isReading) {
  SmallVector<OpOperand *> result;
  for (OpOperand &bufferUse : buffer.getUses()) {
    if (isReading == isBufferReadingOpOperand(bufferUse))
      result.emplace_back(&bufferUse);
  }
  return result;
}

SmallVector<OpOperand *, 1> M::MGP::getInputChainOpOperands(Operation *op) {
  SmallVector<OpOperand *, 1> result;
  for (auto &operand : op->getOpOperands()) {
    if (isa<MEF::ChainType>(operand.get().getType()))
      result.emplace_back(&operand);
  }
  return result;
}

Operation *M::MGP::getUniqueWriter(Value buffer) {
  auto writingOpOperands = allUsingOpOperands(buffer, /*isReading=*/false);
  if (writingOpOperands.size() != 1)
    return nullptr;
  return writingOpOperands.front()->getOwner();
}

Operation *M::MGP::earliestUsingOp(mlir::ResultRange results) {
  Operation *earliest = nullptr;
  for (auto result : results) {
    for (auto user : result.getUsers()) {
      if (earliest == nullptr || user->isBeforeInBlock(earliest))
        earliest = user;
    }
  }
  return earliest;
}

void M::MGP::moveAllBefore(Operation *origTargetOp, Operation *origAnchorOp) {
  std::deque<std::pair<Operation *, Operation *>> worklist;
  worklist.emplace_back(origTargetOp, origAnchorOp);
  while (!worklist.empty()) {
    auto [targetOp, anchorOp] = worklist.front();
    worklist.pop_front();
    if (targetOp == nullptr || anchorOp == nullptr)
      continue;
    if (targetOp->getBlock() != anchorOp->getBlock())
      continue;
    if (targetOp->isBeforeInBlock(anchorOp))
      continue;
    targetOp->moveBefore(anchorOp);
    for (auto operand : targetOp->getOperands())
      worklist.emplace_front(operand.getDefiningOp(), targetOp);
  }
}

/// If chain is defined by a grt.chain.create operation then accumulate
/// all of its operands into primitiveChains. Otherwise adds chain directly
/// to primitiveChains.
void M::MGP::accumulatePrimitiveChains(ValueSetVector &primitiveChains,
                                       Value chain) {
  if (auto chainCreateOp = chain.getDefiningOp<GRT::ChainCreateOp>()) {
    for (auto operand : chainCreateOp.getOperands())
      accumulatePrimitiveChains(primitiveChains, operand);
  } else {
    primitiveChains.insert(chain);
  }
}

/// As for waitForChains, but reuse existingUniqueChainCreateOp instead of
/// creating a new op if it is non-null.
static Value
waitForChainsReusing(OpBuilder &builder, mlir::Location loc,
                     GRT::ChainCreateOp existingUniqueChainCreateOp,
                     ArrayRef<Value> chains) {
  assert(!chains.empty());

  if (chains.size() == 1)
    return chains.front();

  if (existingUniqueChainCreateOp) {
    // Just update the operands to the existing merge op.
    existingUniqueChainCreateOp->setOperands(chains);
    return existingUniqueChainCreateOp.getResult();
  }

  return builder.create<GRT::ChainCreateOp>(loc, chains).getResult();
}

Value M::MGP::waitForChains(OpBuilder &builder, mlir::Location loc,
                            ArrayRef<Value> chains) {
  return waitForChainsReusing(builder, loc, /*existingUniqueChainCreateOp=*/{},
                              chains);
}

LogicalResult M::MGP::alsoWaitForChains(OpBuilder &builder, Operation *op,
                                        ArrayRef<Value> additionalChains) {
  if (additionalChains.empty())
    return success();

  auto existingInChainOpOperands = getInputChainOpOperands(op);
  if (existingInChainOpOperands.empty())
    return failure();

  for (OpOperand *inChainOpOperand : existingInChainOpOperands) {
    Value inChain = inChainOpOperand->get();

    // Though the SimplifyChainsPass and CSE can clean things up, a few
    // simple syntactic tests can avoid unnecessary changes.
    ValueSetVector existingPrimitiveChains;
    accumulatePrimitiveChains(existingPrimitiveChains, inChain);
    ValueSetVector newPrimitiveChains;
    for (auto additionalChain : additionalChains)
      accumulatePrimitiveChains(newPrimitiveChains, additionalChain);

    if (llvm::set_is_subset(newPrimitiveChains, existingPrimitiveChains)) {
      // Nothing to change.
      continue;
    }

    if (llvm::set_is_subset(existingPrimitiveChains, newPrimitiveChains) &&
        additionalChains.size() == 1) {
      // Just switch to using the additional chain.
      op->setOperand(inChainOpOperand->getOperandNumber(),
                     additionalChains.front());
      continue;
    }

    // No luck -- we'll need to explicitly merge the existing and new chains.
    // This may result in duplicate merges for the same set of primitive
    // chains, and merges over redundant changes. The SimplifyChainsPass and
    // CSE will clean these up.
    // NOTE: Can't use llvm::set_union here since SetVector::insert returns
    // a bool rather than (itr, bool) pair.
    for (Value newChain : newPrimitiveChains)
      existingPrimitiveChains.insert(newChain);

    // Is inChain the only use of a grt.chain.create operation?
    // If so we can modify that operation directly instead of creating a
    // new operation.
    GRT::ChainCreateOp existingUniqueChainCreateOp = nullptr;
    if (inChain.hasOneUse()) {
      if (auto chainCreateOp = inChain.getDefiningOp<GRT::ChainCreateOp>())
        existingUniqueChainCreateOp = chainCreateOp;
    }

    // Wait on the merge of the accumulated primitive chains.
    auto insertPoint = builder.saveInsertionPoint();
    builder.setInsertionPoint(op);
    Value newInChain =
        waitForChainsReusing(builder, op->getLoc(), existingUniqueChainCreateOp,
                             existingPrimitiveChains.takeVector());
    builder.restoreInsertionPoint(insertPoint);
    op->setOperand(inChainOpOperand->getOperandNumber(), newInChain);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// InitState
//===----------------------------------------------------------------------===//

InitState::InitState(MLIRContext *context, Block *init)
    : init(init), builder(context) {
  init->walk([this](ContextCreateOp op) { contextCreateOp = op; });
  assert(contextCreateOp &&
         "model is missing mgp.context.create op in init block");
  builder.setInsertionPoint(&init->back());
}

/// 'Allocates' a slot in the context.
size_t InitState::allocSlot() {
  size_t slot = contextCreateOp.getNumSlots();
  contextCreateOp.setNumSlots(slot + 1);
  return slot;
}

/// Returns the current final 'init' chain result.
Value InitState::getFinalChain() { return init->back().getOperand(0); }

/// Update the final 'init' chain result.
void InitState::updateFinalChain(Value newChain) {
  init->back().setOperand(0, newChain);
};

/// Returns the 'init' block's context.
Value InitState::getContext() { return contextCreateOp.getResult(); }

//===----------------------------------------------------------------------===//
// ExecuteState
//===----------------------------------------------------------------------===//

ExecuteState::ExecuteState(MLIRContext *context, Block *execute)
    : execute(execute), builder(context) {
  assert(execute->getNumArguments() >= 2);
  assert(isa<MEF::ChainType>(execute->getArgument(0).getType()));
  assert(isa<ContextType>(execute->getArgument(1).getType()));
}

/// Returns 'root' chain, ie the first argument to the 'execute' block.
/// If any non-root chain is ready then the root chain is guaranteed to be
/// ready, and can thus be ignored.
Value ExecuteState::getRootChain() const { return execute->getArgument(0); }

/// Returns MGP context argument to the 'execute' block.
Value ExecuteState::getContext() const { return execute->getArgument(1); }

/// Returns the additional arguments to 'execute' block
iterator_range<mlir::BlockArgument *> ExecuteState::getExtraArgs() const {
  return llvm::drop_begin(execute->getArguments(), 2);
}
