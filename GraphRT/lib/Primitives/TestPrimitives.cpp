//===- TestPrimitives.cpp -------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'test' dialect primitives provided by the GraphRT
// library.  These are intended for writing executor testcases.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFExecutor/PrimitiveFrame.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "GraphRT/Primitives.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "llvm/ADT/StringRef.h"
using namespace M;

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//

static Chain print_hello(Chain chain) {
  printf("hello bef executor!\n");
  fflush(stdout);
  return chain;
}
static void memory_leak_one_int32(Runtime &runtime) {
  // We allocate an integer to intentionally cause a memory leak for testing.
  runtime.allocate<int32_t>();
}

//===----------------------------------------------------------------------===//
// Error handling
//===----------------------------------------------------------------------===//

// This produces only failure/success as Error.
static ErrorOr<int32_t> fail() { return Error("something bad happened"); }

// This produces a normal output and an error output.
static std::pair<ErrorOr<int32_t>, ErrorOr<int32_t>> partial_fail() {
  return {42, Error("something partially bad happened")};
}

//===----------------------------------------------------------------------===//
// Cancellation
//===----------------------------------------------------------------------===//

// Primitive for testing cancellation logic.
static Chain cancel(Chain inChain, Runtime &runtime) {
  // Do NOT do this in a normal primitive. Runtime::cancelExecution() should
  // only be called by clients external to the BEFExecutor.
  runtime.cancelExecution("Cancelled!");
  return inChain;
}

//===----------------------------------------------------------------------===//
// Async Operations
//===----------------------------------------------------------------------===//

static AsyncValueRef<int32_t> async_add_i32(int32_t lhs, int32_t rhs,
                                            Runtime &runtime) {
  // Even though a single scalar add is trivial, we can do it on a background
  // thread if we'd like!
  return addTask(runtime, [lhs, rhs] { return lhs + rhs; });
}

/// This registers the 'test' dialect primitives, intended for writing executor
/// testcases.
void M::registerGraphRTTestPrimitives(PrimitiveRegistry &registry) {
  registry.add("test.print_hello", M_PRIMITIVE(print_hello));
  registry.add("test.memory_leak_one_int32",
               M_PRIMITIVE(memory_leak_one_int32));

  // Error handling.
  registry.add("test.fail", M_PRIMITIVE(fail));
  registry.add("test.partial_fail", M_PRIMITIVE(partial_fail));

  // Cancellation.
  registry.add("test.cancel", M_PRIMITIVE(cancel));

  registry.add("test.async.add.i32", M_PRIMITIVE(async_add_i32));
}
