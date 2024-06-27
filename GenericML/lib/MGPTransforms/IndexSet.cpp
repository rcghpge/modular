//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/IndexSet.h"
#include "GenericML/MGPTransforms/Utils.h"
#include "GraphRT/GRTDialect/GRTControlFlowInterface.h"
#include "Support/AssertStream.h"

using namespace M;
using namespace M::MGP;

void M::MGP::visitValues(IndexSet<Value> &indexSet, Block *block) {
  size_t blockIndex = indexSet.blockIndexMap.size();
  indexSet.blockIndexMap.insert({block, blockIndex});
  for (Value arg : block->getArguments())
    indexSet.addV(arg);
  for (Operation &op : block->getOperations()) {
    if (isa<GRT::GRTControlFlowInterface>(op))
      for (Region &region : op.getRegions())
        visitValues(indexSet, &region.front());
    for (Value result : op.getResults())
      indexSet.addV(result);
  }
}

void M::MGP::printVImpl(IndexSet<Value> &indexSet, llvm::raw_ostream &os,
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
    for (OpOperand &operand : op.getOpOperands())
      indexSet.addV(&operand);
    if (isa<GRT::GRTControlFlowInterface>(op))
      for (Region &region : op.getRegions())
        visitOperands(indexSet, &region.front());
  }
}

IndexSet<const OpOperand *> M::MGP::createOperandIndexSet(ModelOp model) {
  IndexSet<const OpOperand *> indexSet(model);
  visitOperands(indexSet, model.getInitBlock());
  visitOperands(indexSet, model.getExecuteBlock());
  return indexSet;
}

void M::MGP::printVImpl(IndexSet<const OpOperand *> &indexSet,
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
    // TODO: Avoid scan.
    size_t occurrenceIndex = 0;
    for (const OpOperand *otherOperand : indexSet.allVs) {
      if (otherOperand == indexSet.distinguished)
        continue;
      if (otherOperand == operand)
        break;
      if (otherOperand->get() == value)
        ++occurrenceIndex;
    }
    os << "[" << blockItr->second << ", " << occurrenceIndex << "]";
  }
}

void M::MGP::visitOperations(IndexSet<Operation *> &indexSet, Block *block) {
  size_t blockIndex = indexSet.blockIndexMap.size();
  indexSet.blockIndexMap.insert({block, blockIndex});
  for (Operation *op : getDataflowOrder(block)) {
    indexSet.addV(op);
    if (isa<GRT::GRTControlFlowInterface>(op))
      for (Region &region : op->getRegions())
        visitOperations(indexSet, &region.front());
  }
}

IndexSet<Operation *> M::MGP::createOperationIndexSet(ModelOp model) {
  IndexSet<Operation *> indexSet(model);
  visitOperations(indexSet, model.getInitBlock());
  visitOperations(indexSet, model.getExecuteBlock());
  return indexSet;
}

void M::MGP::printVImpl(IndexSet<Operation *> &indexSet, llvm::raw_ostream &os,
                        Operation *op) {
  if (isa<MGP::ModelOp>(op)) {
    // This can happen because we sometimes take the parent of an op.
    os << "MODEL";
    return;
  }
  if (op == indexSet.distinguished) {
    os << "DIST";
    return;
  }
  auto itr = indexSet.blockIndexMap.find(op->getBlock());
  assert(itr != indexSet.blockIndexMap.end());
  if (op->getNumResults() == 0) {
    if (isa<MEF::OutputOp>(op))
      os << "out";
    else
      // Use the op name even though may not be disambiguated by the block
      // index alone.
      os << op->getName();
  } else {
    // Represent the operation by the pretty printed form of it's first result
    Value result = op->getResult(0);
    result.printAsOperand(os, mlir::OpPrintingFlags().printGenericOpForm());
  }
  if (op->getBlock() != indexSet.model.getExecuteBlock())
    os << "[" << itr->second << "]";
}
