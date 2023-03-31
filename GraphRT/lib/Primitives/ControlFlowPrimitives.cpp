//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements generic control flow primitives for the executor.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/Primitives.h"

#include "GraphRT/MEFExecutor/MEFInvokable.h"
#include "GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "LLCL/ForwardDecls.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"
using namespace M;

/// Return a new chain that is ready.
static AsyncValueRef<Chain> chain_create(Runtime &runtime) {
  return runtime.getReadyChain().copy();
}

/// Strictly wait for all the chain operands, and return one when they're ready.
static AsyncValueRef<Chain> chain_merge(Runtime &runtime,
                                        RemainingArguments arguments) {
  if (!arguments.empty())
    return std::move(arguments[0]);
  return runtime.getReadyChain().copy();
}

static void invoke(RemainingArguments arguments, RemainingResults results,
                   MEFAttr<MEFSymbolRef> targetName, MEFInvokable *target,
                   Runtime &runtime) {
  // We don't use the attrs passed in - it's just the symbol name.
  assert(target->getNumArgs() == arguments.size() && "argument count mismatch");
  assert(target->getNumResults() == results.size() && "result count mismatch");
  target->invoke(arguments.get(), results.get(), runtime);
}

//===----------------------------------------------------------------------===//
// grt.if
//===----------------------------------------------------------------------===//

/// This primitive dispatches to a 'true' or 'false' invokable based on a
/// condition.
///
/// Arguments: The first argument is the condition, with type i1/bool, and any
/// additional arguments are passed to the selected invokable.
///
/// trueInv/FalseInv:  The invokables must have matching signatures, and their
/// signatures must match grt.if's signature, exempting the extra i1 for the
/// condition.
static void grt_if(AsyncValueRef<bool> cond, RemainingArguments args,
                   RemainingResults results, MEFInvokable *trueInv,
                   MEFInvokable *falseInv, Runtime &runtime) {
  assert(trueInv->getNumArgs() == args.size() && "argument count mismatch");
  assert(trueInv->getNumResults() == results.size() && "result count mismatch");
  assert(trueInv->getNumArgs() == falseInv->getNumArgs() &&
         trueInv->getNumResults() == falseInv->getNumResults() &&
         "true and false invokable types need to line up");

  // If the condition is already resolved and non-error, then avoid some
  // overhead.
  if (cond.getPointer()->isValueAvailable()) {
    MEFInvokable *bodyInv = *cond ? trueInv : falseInv;
    bodyInv->invoke(args.get(), results.get(), runtime);
    return;
  }

  // Otherwise, go full async.  Define results as IndirectAsync values.
  results.allocateIndirectResults();

  struct IfState {
    RCRef<MEFInvokable> trueInv, falseInv;
    SmallVector<AnyAsyncValueRef> operands;
    SmallVector<AnyAsyncValueRef> results;

    /// This method is invoked when the AsyncValue for `cond` resolves.  The
    /// rest of the operands may still be unavailable, but results are filled
    /// in with the IndirectAsyncValue results to fill in.
    void runWhenCondResolves(AsyncValueRef<bool> cond) {
      // If we have an error, then we can force propagate errors to all the
      // results.
      if (cond.isError()) {
        for (AnyAsyncValueRef &result : results)
          std::move(result).resolveIndirect(cond.copy());
        return;
      }

      // Otherwise, execute the correct invokable based on the cond.
      MEFInvokable *bodyInv = (*cond ? trueInv : falseInv).getPointer();
      SmallVector<AnyAsyncValueRef> bodyResults;
      bodyResults.resize(results.size());
      bodyInv->invoke(operands, bodyResults, cond.getRuntime());

      // We complete by passing the final result values to the output of the
      // repeat loop overall.
      for (size_t resNo = 0, e = bodyResults.size(); resNo != e; ++resNo)
        std::move(results[resNo])
            .resolveIndirect(std::move(bodyResults[resNo]));
    }
  };

  IfState state = {copyRCRef(trueInv), copyRCRef(falseInv), args.takeAsVector(),
                   results.copyAsVector()};

  std::move(cond).andThenSync(
      [state = std::move(state)](AsyncValueRef<bool> &&cond) mutable {
        state.runWhenCondResolves(std::move(cond));
      });
}

//===----------------------------------------------------------------------===//
// grt.repeat.i32
//===----------------------------------------------------------------------===//

/// "Repeat N times" loop.  This takes a single int32_T iteration count, plus
/// arguments that are passed to the body and eventually returned.
///
/// This takes the count as an AsyncValue because this is a nonstrict primitive,
/// and the count may not be available yet.
static void repeat_i32(AsyncValueRef<int32_t> count, RemainingArguments args,
                       RemainingResults results, MEFInvokable *body,
                       Runtime &runtime) {
  assert(args.size() == results.size() && args.size() == body->getNumArgs() &&
         results.size() == body->getNumResults());

  // Define results as IndirectAsync values.  The actual results are set in the
  // last iteration of the loop.
  results.allocateIndirectResults();

  /// Capture and add-ref each argument and result AsyncValue into a state we
  /// can manage.
  struct RepeatState {
    Runtime &runtime;
    RCRef<MEFInvokable> invokable;
    SmallVector<AnyAsyncValueRef> operands;
    SmallVector<AnyAsyncValueRef> repeatResults;

    /// This is the number of iterations left to run.
    int32_t iterationsLeft;

    /// This method is invoked when the AsyncValue for count resolves.  The rest
    /// of the operands may still be unavailable, but repeatResults are filled
    /// in with the IndirectAsyncValue results returned from the repeat loop
    /// overall.
    void runWhenCountResolves(const AsyncValueRef<int32_t> &count) {
      // If we have an error, then we can force propagate errors to all the
      // results.
      if (count.isError()) {
        for (AnyAsyncValueRef &result : repeatResults)
          std::move(result).resolveIndirect(count.copy());
        return;
      }

      // Now that count is resolved, we know how many iterations to do.
      iterationsLeft = *count;

      // Kick off the next iteration of the loop on another task.
      addTask(runtime,
              [state = std::move(*this)]() mutable { state.doIteration(); });
    }

    void doIteration() {
      // If there are no iterations left, then the repeat results are
      // resolved to the outputs of the last iteration and we're done.
      if (iterationsLeft == 0) {
        // We complete by passing the final result values to the output of the
        // repeat loop overall.
        for (size_t argNo = 0, e = operands.size(); argNo != e; ++argNo)
          std::move(repeatResults[argNo])
              .resolveIndirect(std::move(operands[argNo]));
        return;
      }

      // If the execution is canceled, give up.
      if (AsyncValue *cancelValue = runtime.getCancelValue()) {
        for (auto &result : repeatResults)
          std::move(result).resolveIndirect(copyRCRef(cancelValue));
        return;
      }

      // Otherwise, we have an iteration to perform.
      --iterationsLeft;

      SmallVector<AnyAsyncValueRef> bodyResults;
      bodyResults.resize(repeatResults.size());
      invokable->invoke(operands, bodyResults, runtime);

      // When all the results of the previous computation complete, kick off the
      // next iteration.  Note that this is a policy decision - we could just
      // "unroll the loop" completely, which would increase parallelism.  This
      // is conservatively correct though.
      andThenSyncMoving(
          bodyResults, [state = std::move(*this)](
                           MutableArrayRef<AnyAsyncValueRef> values) mutable {
            // bodyResults become the operands of the next iteration.
            for (auto [operand, newValue] : llvm::zip(state.operands, values))
              operand = std::move(newValue);

            addTask(state.runtime, [state = std::move(state)]() mutable {
              state.doIteration();
            });
          });
    }
  };

  RepeatState state = {runtime, copyRCRef(body), args.takeAsVector(),
                       results.copyAsVector(), 0};

  // Dispatch when the `count` becomes available (possibly immediately).
  std::move(count).andThenSync(
      [state = std::move(state)](AnyAsyncValueRef &&count) mutable {
        state.runWhenCountResolves(std::move(count));
      });
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// This registers the 'graphrt' dialect primitives for control flow.
void M::registerGraphRTControlFlowPrimitives(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "grt.chain.create", chain_create);
  ADD_PRIMITIVE(registry, "grt.chain.merge", chain_merge);
  ADD_PRIMITIVE(registry, "grt.invoke", invoke);
  ADD_PRIMITIVE(registry, "grt.if", grt_if);
  ADD_PRIMITIVE(registry, "grt.repeat.i32", repeat_i32);
}
