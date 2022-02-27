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
#include "LLCL/Runtime/AsyncValueRef.h"
#include "llvm/ADT/StringRef.h"
using namespace M;

// TODO: These should be moved out to a test library when we have more infra.
struct Chain {};

static void test_chain_new(PrimitiveFrame &frame) {
  assert(frame.getNumArgs() == 0 && frame.getNumResults() == 1);
  frame.setResult(
      0, AsyncValueRef<Chain>::createReady(frame.getRuntime(), Chain()));
}

static void test_print_hello(PrimitiveFrame &frame) {
  assert(frame.getNumArgs() == 1 && frame.getNumResults() == 1);
  printf("hello bef executor!\n");
  fflush(stdout);
  frame.setResult(0, copyRCRef(frame.getArg(0)));
}

static void test_constant42_i32(PrimitiveFrame &frame) {
  assert(frame.getNumArgs() == 0 && frame.getNumResults() == 1);
  frame.setResult(0,
                  AsyncValueRef<int32_t>::createReady(frame.getRuntime(), 42));
}

static void test_print_i32(PrimitiveFrame &frame) {
  assert(frame.getNumArgs() == 2 && frame.getNumResults() == 1);
  printf("int32 = %d\n", frame.getArg<int32_t>(0));
  fflush(stdout);
  frame.setResult(0, copyRCRef(frame.getArg(1)));
}

static void test_add_i32(PrimitiveFrame &frame) {
  assert(frame.getNumArgs() == 2 && frame.getNumResults() == 1);
  int32_t result = frame.getArg<int32_t>(0) + frame.getArg<int32_t>(1);
  frame.setResult(
      0, AsyncValueRef<int32_t>::createReady(frame.getRuntime(), result));
}

/// This registers the 'test' dialect primitives, intended for writing executor
/// testcases.
void M::registerGraphRTTestPrimitives(PrimitiveRegistry &registry) {
  registry.add("test.chain.new", test_chain_new);
  registry.add("test.print_hello", test_print_hello);
  registry.add("test.constant42.i32", test_constant42_i32);
  registry.add("test.print.i32", test_print_i32);
  registry.add("test.add.i32", test_add_i32);
}
