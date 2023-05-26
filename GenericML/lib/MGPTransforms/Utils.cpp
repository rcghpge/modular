//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "./Utils.h"

#include "GenericML/MGPDialect/MGPUtils.h"

using namespace M;
using namespace M::MGP;

void M::MGP::remove(SmallVectorImpl<Value> &chains, Value chain) {
  auto itr = llvm::find(chains, chain);
  if (itr != chains.end())
    chains.erase(itr);
}

void M::MGP::insert(SmallVectorImpl<Value> &chains, Value chain) {
  auto itr = llvm::find(chains, chain);
  if (itr == chains.end())
    chains.emplace_back(chain);
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

/// Schedule op for later erasure. (We can't always erase eagerly if
/// we're in the middle of, eg, an iteration over uses/users.)
void ExecuteState::scheduleErase(Operation *op) { toErase.insert(op); }

/// Erase all the 'toErase' entries.
void ExecuteState::erasePending() {
  for (Operation *op : toErase)
    op->erase();
  toErase.clear();
}

/// If chain is defined by a grt.chain.merge operation then accumulate
/// all of its operands into primitiveChains. Otherwise adds chain directly
/// to primitiveChains. The 'root' chain is ignored and will never be
/// accumulated, even if chain is the 'root' chain exactly.
void ExecuteState::accumulatePrimitiveChains(
    SmallVectorImpl<Value> &primitiveChains, Value chain) {
  if (chain == getRootChain())
    return;
  if (auto chainMergeOp = chain.getDefiningOp<GRT::ChainMergeOp>()) {
    for (auto operand : chainMergeOp.getOperands())
      accumulatePrimitiveChains(primitiveChains, operand);
  } else {
    insert(primitiveChains, chain);
  }
}

/// Returns a chain which will only be ready when all of the given chains
/// are ready. Understands a few special cases:
///  - The 'root' chain does not need to be waited on if we are waiting on
///    any other chain.
///  - Only need a merge operation if two or more chains to wait on.
///  - Don't need to wait on the same chain more than once.
///
/// If existingUniqueChainMergeOp is non-null it will be mutated to construct
/// the result.
Value ExecuteState::mergeChains(GRT::ChainMergeOp existingUniqueChainMergeOp,
                                SmallVector<Value> &chains,
                                Operation *consumingOp) {
  if (chains.empty())
    return getRootChain();

  if (chains.size() == 1)
    return *chains.begin();

  // Don't bother waiting on the root chain.
  remove(chains, getRootChain());

  if (chains.size() == 1)
    return *chains.begin();

  if (existingUniqueChainMergeOp) {
    // Just update the operands to the existing merge op.
    existingUniqueChainMergeOp->setOperands(chains);
    return existingUniqueChainMergeOp.getResult();
  }

  // Construct a new merge op.
  builder.setInsertionPoint(consumingOp);
  return builder
      .create<GRT::ChainMergeOp>(consumingOp->getLoc(),
                                 /*result=*/getRootChain().getType(),
                                 /*operands=*/chains)
      .getResult();
}

/// Ensures the input chain for op (if any) is ready only when
/// additionalChain is also ready.
///
/// Return failure if `op` doesn't have any input chain to merge with
/// a non-root `additionalChain`.
LogicalResult ExecuteState::alsoWaitForChain(Operation *op,
                                             Value additionalChain) {
  if (additionalChain == getRootChain())
    // Special case: don't wait for the root chain in addition.
    return success();

  bool hasInputChain = false;
  for (OpOperand *inChainOpOperand : getInputChainOpOperands(op)) {
    hasInputChain = true;
    Value inChain = inChainOpOperand->get();
    if (inChain == getRootChain()) {
      // Special case: just switch if currently waiting on the root chain.
      op->setOperand(inChainOpOperand->getOperandNumber(), additionalChain);
    } else if (inChain == additionalChain) {
      // Special case: no-op
      /* no-op */
    } else {
      // Is inChain the only use of an grt.chain.merge operation?
      // If so we can modify that operation directly instead of creating a
      // new operation.
      GRT::ChainMergeOp existingUniqueChainMergeOp = nullptr;
      if (auto chainMergeOp = inChain.getDefiningOp<GRT::ChainMergeOp>()) {
        if (chainMergeOp->getBlock() == op->getBlock()) {
          auto itr = chainMergeOp->getUsers().begin();
          if (*itr == op && ++itr == chainMergeOp->getUsers().end())
            existingUniqueChainMergeOp = chainMergeOp;
        }
      }

      // Accumulate the primitive chains to wait on from
      // inChain and additionalChain.
      SmallVector<Value> primitiveChains;
      accumulatePrimitiveChains(primitiveChains, inChain);
      accumulatePrimitiveChains(primitiveChains, additionalChain);

      /// TODO(#7481):
      ///  - Simplify primitiveChains using a dominates relation. Ie avoid
      ///    depending on both %x and %y if ready(%x) implies ready(%y).
      ///  - Avoid creating the duplicate merge operations with the same set
      ///    of primitive chains.
      ///  - Handle all of this with control flow ops and their isolated
      ///    regions. In particular, the 'root' chain can end up being passed
      ///    into a control flow region but its special status is forgotten.

      // Wait on the merge of the accumulated primitive chains.
      Value newInChain =
          mergeChains(existingUniqueChainMergeOp, primitiveChains, op);
      op->setOperand(inChainOpOperand->getOperandNumber(), newInChain);
    }
  }

  return success(hasInputChain);
}

//===----------------------------------------------------------------------===//
// ValueTestPrinter
//===----------------------------------------------------------------------===//

ValueTestPrinter::ValueTestPrinter(ModelOp model) : model(model) {
  blockMap.insert({model.getExecuteBlock(), blockMap.size()});
  for (auto arg : model.getExecuteBlock()->getArguments())
    allValues.emplace_back(arg);
  model.getExecuteBlock()->walk<mlir::WalkOrder::PostOrder>([&](Operation *op) {
    for (auto result : op->getResults())
      allValues.emplace_back(result);
    for (auto &region : op->getRegions()) {
      Block *block = &region.getBlocks().front();
      blockMap.insert({block, blockMap.size()});
      for (auto arg : region.getBlocks().front().getArguments())
        allValues.emplace_back(arg);
    }
  });
}

void ValueTestPrinter::printModel(llvm::raw_ostream &os) {
  model.print(os, mlir::OpPrintingFlags().printGenericOpForm());
  os << "\n\n";
}

void ValueTestPrinter::printValue(llvm::raw_ostream &os, Value value) {
  value.printAsOperand(os, mlir::OpPrintingFlags().printGenericOpForm());
  size_t blockId = blockMap.lookup(value.getParentBlock());
  if (blockId)
    os << "[" << blockId << "]";
}

void ValueTestPrinter::printPredicate(llvm::raw_ostream &os, StringRef predName,
                                      const std::function<bool(Value)> &pred) {
  for (size_t i = 0, n = allValues.size(); i < n; ++i) {
    Value value = allValues[i];
    if (pred(value)) {
      os << predName << "(";
      printValue(os, value);
      os << ")\n";
    }
  }
}

void ValueTestPrinter::printRelation(
    llvm::raw_ostream &os, StringRef relName,
    const std::function<bool(Value, Value)> &rel) {
  for (size_t i = 0, n = allValues.size(); i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      Value lhs = allValues[i];
      Value rhs = allValues[j];
      if (rel(lhs, rhs)) {
        os << relName << "(";
        printValue(os, lhs);
        os << ", ";
        printValue(os, rhs);
        os << ")\n";
      }
    }
  }
}
