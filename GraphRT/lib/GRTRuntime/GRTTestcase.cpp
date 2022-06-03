//===- GRTTestcase.cpp ----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/GRTTestcase.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"

using namespace M;

GRTTestcase::GRTTestcase(const PrimitiveDescriptor &op)
    : name(*BEFAttr<StringRef>(op.getRawAttr(0))), body(op.getInvokable(0)) {
  assert(op.getNumInvokables() == 1 &&
         "cannot construct a BEFInvokablePrimitive from a TopLevelOp that has "
         "more than one primitive");
}

StringLiteral GRTTestcase::getEntityName() { return "grt.testcase"; }
