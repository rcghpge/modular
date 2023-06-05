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
