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
#include "LLCL/ForwardDecls.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/Chain.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/StringRef.h"
using namespace M;

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//

static Chain test_print_hello(Chain chain) {
  printf("hello bef executor!\n");
  fflush(stdout);
  return chain;
}

static void test_constant_i32(PrimitiveFrame &frame) {
  assert(frame.getNumArgs() == 0 && frame.getNumResults() == 1 &&
         frame.getNumAttrs() == 1);
  frame.emplaceResult<int32_t>(0, frame.getAttr<int32_t>(0));
}

static Chain test_print_i32(int32_t value, Chain chain) {
  printf("int32 = %d\n", value);
  fflush(stdout);
  return chain;
}

static int32_t test_add_i32(int32_t lhs, int32_t rhs) { return lhs + rhs; }

//===----------------------------------------------------------------------===//
// Error handling
//===----------------------------------------------------------------------===//

// This produces only failure/success as Error.
static ErrorOr<int32_t> test_fail() { return Error("something bad happened"); }

// This produces a normal output and an error output.
static std::pair<ErrorOr<int32_t>, ErrorOr<int32_t>> test_partial_fail() {
  return {42, Error("something partially bad happened")};
}

//===----------------------------------------------------------------------===//
// Async Operations
//===----------------------------------------------------------------------===//

static AsyncValueRef<int32_t> test_async_add_i32(Runtime *runtime, int32_t lhs,
                                                 int32_t rhs) {
  // Even though a single scalar add is trivial, we can do it on a background
  // thread if we'd like!
  return addTask(*runtime, [lhs, rhs] { return lhs + rhs; });
}

/// This registers the 'test' dialect primitives, intended for writing executor
/// testcases.
void M::registerGraphRTTestPrimitives(PrimitiveRegistry &registry) {
  registry.add("test.print_hello", M_PRIMITIVE(test_print_hello));
  registry.add("test.constant.i32", test_constant_i32);
  registry.add("test.print.i32", M_PRIMITIVE(test_print_i32));
  registry.add("test.add.i32", M_PRIMITIVE(test_add_i32));

  registry.add("test.fail", M_PRIMITIVE(test_fail));
  registry.add("test.partial_fail", M_PRIMITIVE(test_partial_fail));

  registry.add("test.async.add.i32", M_PRIMITIVE(test_async_add_i32));
}
