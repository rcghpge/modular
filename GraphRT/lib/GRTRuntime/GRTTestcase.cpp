//===- GRTTestcase.cpp ----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/GRTTestcase.h"
#include "GraphRT/BEFExecutor/TopLevelEntity.h"

using namespace M;

GRTTestcase::GRTTestcase(const M::TopLevelEntity &op)
    : GRTTestcase(op.getSymbolName(), op.getInvokable(0)) {
  assert(op.getPrimitiveName() == getEntityName() &&
         "wrong kind of top level op");
  assert(op.getNumInvokables() == 1 &&
         "cannot construct a BEFInvokablePrimitive from a TopLevelOp that has "
         "more than one primitive");
}

StringLiteral GRTTestcase::getEntityName() { return "grt.testcase"; }
