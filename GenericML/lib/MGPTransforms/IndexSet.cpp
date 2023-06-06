//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "./IndexSet.h"

using namespace M;
using namespace M::MGP;

void M::MGP::visitValues(IndexSet<Value> &indexSet, Block *block) {
  size_t blockIndex = indexSet.blockIndexMap.size();
  indexSet.blockIndexMap.insert({block, blockIndex});
  for (Value arg : block->getArguments())
    indexSet.addV(arg);
  for (Operation &op : block->getOperations()) {
    for (Region &region : op.getRegions())
      visitValues(indexSet, &region.front());
    for (Value result : op.getResults())
      indexSet.addV(result);
  }
}

void M::MGP::printV(IndexSet<Value> &indexSet, llvm::raw_ostream &os,
                    Value value) {
  if (value == indexSet.distinguished) {
    os << "DIST";
    return;
  }
  auto itr = indexSet.blockIndexMap.find(value.getParentBlock());
  assert(itr != indexSet.blockIndexMap.end());
  value.printAsOperand(os, mlir::OpPrintingFlags().printGenericOpForm());
  if (value.getParentBlock() != indexSet.model.getExecuteBlock())
    os << "[" << itr->second << "]";
}

IndexSet<Value> M::MGP::createValueIndexSet(ModelOp model) {
  IndexSet<Value> indexSet(model);
  visitValues(indexSet, model.getInitBlock());
  visitValues(indexSet, model.getExecuteBlock());
  return indexSet;
}

void M::MGP::visitOperands(IndexSet<const OpOperand *> &indexSet,
                           Block *block) {
  size_t blockIndex = indexSet.blockIndexMap.size();
  indexSet.blockIndexMap.insert({block, blockIndex});
  for (Operation &op : block->getOperations()) {
    for (Region &region : op.getRegions())
      visitOperands(indexSet, &region.front());
    for (OpOperand &result : op.getOpOperands())
      indexSet.addV(&result);
  }
}

IndexSet<const OpOperand *> M::MGP::createOperandIndexSet(ModelOp model) {
  IndexSet<const OpOperand *> indexSet(model);
  visitOperands(indexSet, model.getInitBlock());
  visitOperands(indexSet, model.getExecuteBlock());
  return indexSet;
}

void M::MGP::printV(IndexSet<const OpOperand *> &indexSet,
                    llvm::raw_ostream &os, const OpOperand *operand) {
  if (operand == indexSet.distinguished) {
    os << "DIST";
    return;
  }
  Value value = operand->get();
  auto blockItr = indexSet.blockIndexMap.find(value.getParentBlock());
  assert(blockItr != indexSet.blockIndexMap.end());
  value.printAsOperand(os, mlir::OpPrintingFlags().printGenericOpForm());
  if (value.getParentBlock() != indexSet.model.getExecuteBlock() ||
      !value.hasOneUse()) {
    auto useItr = value.getUses().begin();
    while (useItr != value.getUses().end() && &*useItr != operand)
      ++useItr;
    assert(useItr != value.getUses().end());
    size_t occurrenceIndex = std::distance(value.getUses().begin(), useItr);
    os << "[" << blockItr->second << ", " << occurrenceIndex << "]";
  }
}
