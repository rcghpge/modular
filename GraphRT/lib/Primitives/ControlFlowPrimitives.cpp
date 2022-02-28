//===- ControlFlowPrimitives.cpp ------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements generic control flow primitives for the executor.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFExecutor/PrimitiveFrame.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "GraphRT/Primitives.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"
using namespace M;

static void bef_chain_new(PrimitiveFrame &frame) {
  assert(frame.getNumArgs() == 0 && frame.getNumResults() == 1);
  frame.setResult(0, frame.getRuntime().getReadyChain());
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// This registers primitives used for core control flow operations.
void M::registerGraphRTControlFlowPrimitives(PrimitiveRegistry &registry) {
  registry.add("bef.chain.new", bef_chain_new);
}
