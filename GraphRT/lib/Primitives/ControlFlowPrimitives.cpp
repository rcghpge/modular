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

/// "Repeat N times" loop.  This takes a single int32_T iteration count, plus
/// arguments that are passed to the bodyFn and eventually returned.
///
/// This takes the count as an AsyncValue because this is a nonstrict primitive,
/// and the count may not be available yet.
static void repeat_i32(AsyncValueRef<int32_t> count, RemainingArguments args,
                       RemainingResults results, const Function *bodyFn,
                       Runtime &runtime) {
  assert(args.size() == results.size() && args.size() == bodyFn->getNumArgs() &&
         results.size() == bodyFn->getNumResults());

  // Define results as IndirectAsync values.  The actual results is set in the
  // last iteration of the loop.
  results.allocateIndirectResults(runtime);

  auto *countPtr = count.getPointer();

  /// Capture and add-ref each argument and result AsyncValue into a state we
  /// can manage.
  struct RepeatState {
    AsyncValueRef<int32_t> count;
    RCRef<const Function> function;
    SmallVector<RCRef<AsyncValue>> operands;
    SmallVector<RCRef<AsyncValue>> repeatResults;

    /// This method is invoked when the AsyncValue for count resolves.  The rest
    /// of the operands may still be unavailable, but repeatResults are filled
    /// in with the IndirectAsyncValue results returned from the repeat loop
    /// overall.
    void runWhenCountResolves() {
      // If we have an error, then we can force propagate errors to all the
      // results.
      if (count->isError()) {
        for (RCRef<AsyncValue> &result : repeatResults)
          result->resolveIndirect(count.copy());
        return;
      }

      // Otherwise, we kick off all of the iterations at once!
      SmallVector<RCRef<AsyncValue>> bodyResults;

      // TODO: We could be smarter about this, and only kick off an iteration
      // when at least one result from the previous iteration is available.  We
      // could also look for cancellation state in `Runtime` which would make
      // cancellation quash computation faster.
      for (int32_t i = 0, e = count->get<int32_t>(); i != e; ++i) {
        bodyResults.resize(repeatResults.size());
        function->execute(operands, bodyResults);
        // bodyResults become the operands of the next iteration.
        operands = std::move(bodyResults);
        bodyResults.clear();
      }

      // We complete by passing the final result values to the output of the
      // repeat loop overall.
      for (size_t argNo = 0, e = operands.size(); argNo != e; ++argNo)
        repeatResults[argNo]->resolveIndirect(std::move(operands[argNo]));
    }
  };

  RepeatState repeatState = {std::move(count), LLCL::copyRCRef(bodyFn), {}, {}};

  repeatState.operands.reserve(args.size());
  repeatState.repeatResults.reserve(results.size());
  for (auto &arg : args.get())
    repeatState.operands.push_back(std::move(arg));
  for (auto &result : results.get())
    repeatState.repeatResults.push_back(result.copy());

  // Dispatch when the `count` becomes available (possibly immediately).
  countPtr->andThen([repeatState = std::move(repeatState)]() mutable {
    repeatState.runWhenCountResolves();
  });
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// This registers the 'graphrt' dialect primitives for control flow.
void M::registerGraphRTControlFlowPrimitives(PrimitiveRegistry &registry) {
  registry.add("grt.chain.new", M_PRIMITIVE(chain_new));
  registry.add("grt.call", M_PRIMITIVE(call));
  registry.add("grt.repeat.i32", M_PRIMITIVE(repeat_i32));
}
