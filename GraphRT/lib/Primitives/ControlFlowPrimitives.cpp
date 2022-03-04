//===- ControlFlowPrimitives.cpp ------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements generic control flow primitives for the executor.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/Primitives.h"

#include "GraphRT/BEFExecutor/Function.h"
#include "GraphRT/BEFExecutor/PrimitiveFrame.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "LLCL/ForwardDecls.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"
using namespace M;

static AsyncValueRef<Chain> graphrt_chain_new(Runtime &runtime) {
  return runtime.getReadyChain().copy();
}

static void graphrt_call(PrimitiveFrame &frame) {
  const Function *fn = frame.getFunction(0);
  assert(fn->getNumArgs() == frame.getNumArgs() && "argument count mismatch");
  assert(fn->getNumResults() == frame.getNumResults() &&
         "result count mismatch");

  fn->execute(frame.getArguments(), frame.getResults());
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// This registers the 'graphrt' dialect primitives for control flow.
void M::registerGraphRTControlFlowPrimitives(PrimitiveRegistry &registry) {
  registry.add("grt.chain.new", M_PRIMITIVE(graphrt_chain_new));
  registry.add("grt.call", graphrt_call);
}
