//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/GraphRT/GRTTestcase.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"

using namespace M;

GRTTestcase::GRTTestcase(const PrimitiveDescriptor &op)
    : name(*op.getAttr<StringRef>(0)), body(op.getInvokable(0)) {
  assert(op.getNumInvokables() == 1 &&
         "cannot construct a MEFInvokablePrimitive from a TopLevelOp that has "
         "more than one primitive");
}

StringLiteral GRTTestcase::getEntityName() { return "grt.testcase"; }
