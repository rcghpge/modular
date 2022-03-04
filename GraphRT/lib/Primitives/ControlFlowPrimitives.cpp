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

static AsyncValueRef<Chain> chain_new(Runtime &runtime) {
  return runtime.getReadyChain().copy();
}

static void call(RemainingArguments arguments, RemainingResults results,
                 const Function *callee) {
  assert(callee->getNumArgs() == arguments.size() && "argument count mismatch");
  assert(callee->getNumResults() == results.size() && "result count mismatch");
  callee->execute(arguments.get(), results.get());
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// This registers the 'graphrt' dialect primitives for control flow.
void M::registerGraphRTControlFlowPrimitives(PrimitiveRegistry &registry) {
  registry.add("grt.chain.new", M_PRIMITIVE(chain_new));
  registry.add("grt.call", M_PRIMITIVE(call));
}
