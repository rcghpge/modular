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

static void call(Runtime &runtime, RemainingArguments arguments,
                 RemainingResults results, const Function *callee) {
  assert(callee->getNumArgs() == arguments.size() && "argument count mismatch");
  assert(callee->getNumResults() == results.size() && "result count mismatch");
  callee->execute(arguments.get(), results.get(), runtime);
}

//===----------------------------------------------------------------------===//
// grt.if
//===----------------------------------------------------------------------===//

/// This primtive dispatches to a 'true' or 'false' function based on a
/// condition.
///
/// Arguments: The first argument is the condition, with type i1/bool, and any
/// additional arguments are passed to the selected function.
///
/// trueFn/FalseFn:  The functions must have matching signatures, and their
/// signatures must match grt.if's signature, exempting the extra i1 for the
/// condition.
static void grt_if(AsyncValueRef<bool> cond, RemainingArguments args,
                   RemainingResults results, const Function *trueFn,
                   const Function *falseFn, Runtime &runtime) {
  assert(trueFn->getNumArgs() == args.size() && "argument count mismatch");
  assert(trueFn->getNumResults() == results.size() && "result count mismatch");
  assert(trueFn->getNumArgs() == falseFn->getNumArgs() &&
         trueFn->getNumResults() == falseFn->getNumResults() &&
         "true and false function types need to line up");

  // If the condition is already resolved and non-error, then avoid some
  // overhead.
  if (cond->isValueAvailable()) {
    const Function *bodyFn = cond.get() ? trueFn : falseFn;
    bodyFn->execute(args.get(), results.get(), runtime);
    return;
  }

  // Otherwise, go full async.  Define results as IndirectAsync values.
  results.allocateIndirectResults(runtime);

  struct IfState {
    AsyncValueRef<bool> cond;
    RCRef<const Function> trueFn, falseFn;
    SmallVector<RCRef<AsyncValue>> operands;
    SmallVector<RCRef<AsyncValue>> results;

    /// This method is invoked when the AsyncValue for `cond` resolves.  The
    /// rest of the operands may still be unavailable, but results are filled
    /// in with the IndirectAsyncValue results to fill in.
    void runWhenCondResolves(Runtime &runtime) {
      // If we have an error, then we can force propagate errors to all the
      // results.
      if (cond->isError()) {
        for (RCRef<AsyncValue> &result : results)
          result->resolveIndirect(cond.copy());
        return;
      }

      // Otherwise, execute the correct function based on the cond.
      const Function *bodyFn = (cond.get() ? trueFn : falseFn).getPointer();
      SmallVector<RCRef<AsyncValue>> bodyResults;
      bodyResults.resize(results.size());
      bodyFn->execute(operands, bodyResults, runtime);

      // We complete by passing the final result values to the output of the
      // repeat loop overall.
      for (size_t resNo = 0, e = bodyResults.size(); resNo != e; ++resNo)
        results[resNo]->resolveIndirect(std::move(bodyResults[resNo]));
    }
  };

  auto *condPtr = cond.getPointer();
  IfState state = {
      std::move(cond), copyRCRef(trueFn), copyRCRef(falseFn), {}, {}};

  state.operands.reserve(args.size());
  for (auto &arg : args.get())
    state.operands.push_back(std::move(arg));

  state.results.reserve(results.size());
  for (auto &result : results.get())
    state.results.push_back(result.copy());

  condPtr->andThen([&runtime, state = std::move(state)]() mutable {
    state.runWhenCondResolves(runtime);
  });
}

//===----------------------------------------------------------------------===//
// grt.repeat.i32
//===----------------------------------------------------------------------===//

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

  // Define results as IndirectAsync values.  The actual results are set in the
  // last iteration of the loop.
  results.allocateIndirectResults(runtime);

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
    void runWhenCountResolves(Runtime &runtime) {
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
        function->execute(operands, bodyResults, runtime);
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

  auto *countPtr = count.getPointer();
  RepeatState state = {std::move(count), copyRCRef(bodyFn), {}, {}};

  state.operands.reserve(args.size());
  for (auto &arg : args.get())
    state.operands.push_back(std::move(arg));
  state.repeatResults.reserve(results.size());
  for (auto &result : results.get())
    state.repeatResults.push_back(result.copy());

  // Dispatch when the `count` becomes available (possibly immediately).
  countPtr->andThen([&runtime, state = std::move(state)]() mutable {
    state.runWhenCountResolves(runtime);
  });
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// This registers the 'graphrt' dialect primitives for control flow.
void M::registerGraphRTControlFlowPrimitives(PrimitiveRegistry &registry) {
  // TODO: Registration of these types should get folded into auto-magic as part
  // of the ADD_PRIMITIVE macro.  Issue #33.
  AsyncValue::registerType<Chain>();
  AsyncValue::registerType<bool>();
  AsyncValue::registerType<int32_t>();

  ADD_PRIMITIVE(registry, "grt.chain.new", chain_new);
  ADD_PRIMITIVE(registry, "grt.call", call);
  ADD_PRIMITIVE(registry, "grt.if", grt_if);
  ADD_PRIMITIVE(registry, "grt.repeat.i32", repeat_i32);
}
