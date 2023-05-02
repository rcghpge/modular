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
    RCRef<MEFInvokable> trueInvokable, falseInvokable;
    SmallVector<AnyAsyncValueRef> operands;
    SmallVector<AnyAsyncValueRef> ifResults;

    /// Emplaces all ifResults with a copy of errorResult.
    void fail(AnyAsyncValueRef &&errorResult) && {
      for (auto &result : ifResults)
        std::move(result).resolveIndirect(errorResult.copy());
    }

    /// This method is invoked when the AsyncValue for `cond` resolves.  The
    /// rest of the operands may still be unavailable, but results are filled
    /// in with the IndirectAsyncValue results to fill in.
    void runWhenCondResolves(AsyncValueRef<bool> &&cond) && {
      // If we have an error, then we can force propagate errors to all the
      // results.
      if (cond.isError())
        return std::move(*this).fail(std::move(cond));

      // Otherwise, execute the correct invokable based on the cond.
      // (Note we can't just pass ifResults for the invokable's results since
      // they are expected to be null).
      MEFInvokable *bodyInvokable =
          (*cond ? trueInvokable : falseInvokable).getPointer();
      SmallVector<AnyAsyncValueRef> bodyResults;
      bodyResults.resize(ifResults.size());
      bodyInvokable->invoke(operands, bodyResults, cond.getRuntime());

      // We complete by passing the final result values to the output of the
      // overall grt.if op.
      for (auto [ifResult, bodyResult] : llvm::zip(ifResults, bodyResults))
        std::move(ifResult).resolveIndirect(std::move(bodyResult));
    }
  };

  IfState state = {copyRCRef(trueInv), copyRCRef(falseInv), args.takeAsVector(),
                   results.copyAsVector()};

  std::move(cond).andThenSync(
      [state = std::move(state)](AsyncValueRef<bool> &&cond) mutable {
        std::move(state).runWhenCondResolves(std::move(cond));
      });
}

//===----------------------------------------------------------------------===//
// grt.while
//===----------------------------------------------------------------------===//

/// "Repeat while true" loop. Evaluate the 'cond' invokable, passing the current
/// value of the while arguments. If true, evaluate the 'body' invokable,
/// updating the current value of the while arguments. If false, return the
/// current value of the while arguments.
static void grt_while(RemainingArguments args, RemainingResults results,
                      MEFInvokable *cond, MEFInvokable *body,
                      Runtime &runtime) {
  assert(args.size() == results.size());
  assert(cond->getNumArgs() == args.size() &&
         cond->getNumResults() == results.size() + 1);
  assert(body->getNumArgs() == args.size() &&
         body->getNumResults() == results.size());

  // Define results as IndirectAsync values.  The actual results are set in the
  // last iteration of the loop.
  results.allocateIndirectResults();

  /// Capture and add-ref each argument and result AsyncValue into a state we
  /// can manage.
  struct WhileState {
    Runtime &runtime;
    RCRef<MEFInvokable> condInvokable;
    RCRef<MEFInvokable> bodyInvokable;
    SmallVector<AnyAsyncValueRef> currentOperands;
    SmallVector<AnyAsyncValueRef> whileResults;

    /// Emplaces all whileResults with a copy of errorResult.
    void fail(AnyAsyncValueRef &&errorResult) && {
      for (auto &result : whileResults)
        std::move(result).resolveIndirect(errorResult.copy());
    }

    /// Emplaces all whileResults with the currentOperands.
    void finish() && {
      for (auto [whileResult, currentOperand] :
           llvm::zip(whileResults, currentOperands))
        std::move(whileResult).resolveIndirect(std::move(currentOperand));
    }

    /// Returns a copy of currentOperands. An invokable is free to take its
    /// arguments, so we peel off a copy before each call to the 'cond' or
    /// 'body' invokables.
    SmallVector<AnyAsyncValueRef> getCurrentOperandsCopy() {
      SmallVector<AnyAsyncValueRef> result;
      result.reserve(currentOperands.size());
      for (const auto &operand : currentOperands)
        result.emplace_back(operand.copy());
      return result;
    }

    /// Evaluate the 'cond' block to decide if we need to execute the 'body'
    /// block (again).
    void evalCond() && {
      // Start executing the 'cond' block, copying the current operands since
      // they may be needed by the 'body' block.
      SmallVector<AnyAsyncValueRef> operands = getCurrentOperandsCopy();
      SmallVector<AnyAsyncValueRef, 1> condResults;
      condResults.resize(whileResults.size() + 1);
      condInvokable->invoke(operands, condResults, runtime);

      // The 'cond' block results (except the last condition result) are the
      // new current operands (though they need not be ready yet).
      for (auto [currentOperand, bodyResult] :
           llvm::zip(currentOperands, llvm::drop_end(condResults)))
        currentOperand = std::move(bodyResult);

      // Wait for the last condition result of the 'cond' block to be ready.
      std::move(condResults.back())
          .andThenSync([state = std::move(*this)](
                           AnyAsyncValueRef &&condResult) mutable {
            // If the execution is canceled, give up.
            if (AsyncValue *cancelValue = state.runtime.getCancelValue())
              return std::move(state).fail(copyRCRef(cancelValue));
            // If the condition yielded an error, give up.
            if (condResult.isError())
              return std::move(state).fail(std::move(condResult));
            // If the condition is false, we are done.
            if (!condResult.get<bool>())
              return std::move(state).finish();

            // Execute the body (and following cond) in a sub-task.
            addTask(state.runtime, [state = std::move(state)]() mutable {
              std::move(state).evalBody();
            });
          });
    }

    /// Evaluate the 'body' block to update the currentOperands
    void evalBody() && {
      // Start executing the 'body' block, consuming the current operands.
      SmallVector<AnyAsyncValueRef> bodyResults;
      bodyResults.resize(whileResults.size());
      bodyInvokable->invoke(currentOperands, bodyResults, runtime);

      // The 'body' block results are the new current operands (though they
      // need not be ready yet).
      for (auto [currentOperand, bodyResult] :
           llvm::zip(currentOperands, bodyResults))
        currentOperand = std::move(bodyResult);

      // And we'll need to go around again to eval the condition.
      // Note that we don't andThenSync on the body results, and instead just
      // let the 'cond' block control execution.
      std::move(*this).evalCond();
    }
  };

  WhileState state = {runtime, copyRCRef(cond), copyRCRef(body),
                      args.takeAsVector(), results.copyAsVector()};

  // Start the ball rolling on a separate task.
  addTask(runtime, [state = std::move(state)]() mutable {
    std::move(state).evalCond();
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
    RCRef<MEFInvokable> bodyInvokable;
    SmallVector<AnyAsyncValueRef> currentOperands;
    SmallVector<AnyAsyncValueRef> repeatResults;

    /// This is the number of iterations left to run.
    int32_t iterationsLeft;

    /// Emplaces all repeatResults with a copy of errorResult.
    void fail(AnyAsyncValueRef &&errorResult) && {
      for (auto &result : repeatResults)
        std::move(result).resolveIndirect(errorResult.copy());
    }

    /// Emplaces all repeatResults with the currentOperands.
    void finish() && {
      for (auto [repeatResult, currentOperand] :
           llvm::zip(repeatResults, currentOperands))
        std::move(repeatResult).resolveIndirect(std::move(currentOperand));
    }

    /// This method is invoked when the AsyncValue for count resolves.  The rest
    /// of the operands may still be unavailable, but repeatResults are filled
    /// in with the IndirectAsyncValue results returned from the repeat loop
    /// overall.
    void runWhenCountResolves(AsyncValueRef<int32_t> &&count) && {
      // If we have an error, then we can force propagate errors to all the
      // results.
      if (count.isError())
        return std::move(*this).fail(std::move(count));

      // Now that count is resolved, we know how many iterations to do.
      iterationsLeft = *count;

      // Start executing the body.
      addTask(runtime, [state = std::move(*this)]() mutable {
        std::move(state).doIteration();
      });
    }

    void doIteration() && {
      // If there are no iterations left, then the repeat results are
      // resolved to the outputs of the last iteration and we're done.
      if (iterationsLeft <= 0)
        // We complete by passing the final result values to the output of the
        // repeat loop overall.
        return std::move(*this).finish();

      // If the execution is canceled, give up.
      if (AsyncValue *cancelValue = runtime.getCancelValue())
        return std::move(*this).fail(copyRCRef(cancelValue));

      // Otherwise, we have an iteration to perform.
      --iterationsLeft;

      SmallVector<AnyAsyncValueRef> bodyResults;
      bodyResults.resize(repeatResults.size());
      bodyInvokable->invoke(currentOperands, bodyResults, runtime);

      // When all the results of the previous computation complete, kick off the
      // next iteration.  Note that this is a policy decision - we could just
      // "unroll the loop" completely, which would increase parallelism.  This
      // is conservatively correct though.
      andThenSyncMoving(
          bodyResults, [state = std::move(*this)](
                           MutableArrayRef<AnyAsyncValueRef> values) mutable {
            // bodyResults become the operands of the next iteration.
            for (auto [currentOperand, value] :
                 llvm::zip(state.currentOperands, values))
              currentOperand = std::move(value);

            addTask(state.runtime, [state = std::move(state)]() mutable {
              std::move(state).doIteration();
            });
          });
    }
  };

  RepeatState state = {runtime, copyRCRef(body), args.takeAsVector(),
                       results.copyAsVector(), 0};

  // Dispatch when the `count` becomes available (possibly immediately).
  std::move(count).andThenSync(
      [state = std::move(state)](AnyAsyncValueRef &&count) mutable {
        std::move(state).runWhenCountResolves(std::move(count));
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
  ADD_PRIMITIVE(registry, "grt.while", grt_while);
  ADD_PRIMITIVE(registry, "grt.repeat.i32", repeat_i32);
}
