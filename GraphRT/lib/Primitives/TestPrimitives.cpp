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
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/Chain.h"
#include "llvm/ADT/StringRef.h"
using namespace M;
using namespace LLCL;

static void test_chain_new(PrimitiveFrame &frame) {
  assert(frame.getNumArgs() == 0 && frame.getNumResults() == 1);
  frame.setResult(0, frame.getRuntime().getReadyChain());
}

static void test_print_hello(PrimitiveFrame &frame) {
  assert(frame.getNumArgs() == 1 && frame.getNumResults() == 1);
  printf("hello bef executor!\n");
  fflush(stdout);
  frame.setResult(0, copyRCRef(frame.getArg(0)));
}

static void test_constant_i32(PrimitiveFrame &frame) {
  assert(frame.getNumArgs() == 0 && frame.getNumResults() == 1 &&
         frame.getNumAttrs() == 1);
  frame.emplaceResult<int32_t>(0, frame.getAttr<int32_t>(0));
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
  frame.emplaceResult<int32_t>(0, result);
}

//===----------------------------------------------------------------------===//
// Error handling
//===----------------------------------------------------------------------===//

// This kernel produces only failure/success as Error.
static void test_fail(PrimitiveFrame &frame) {
  frame.setResult(0, frame.getError("something bad happened"));
}

// This kernel produces a normal output and an error output.
static void test_partial_fail(PrimitiveFrame &frame) {
  assert(frame.getNumArgs() == 0 && frame.getNumResults() == 2);
  frame.emplaceResult<int32_t>(0, 42);
  frame.fail("something partially bad happened");
}

/// This registers the 'test' dialect primitives, intended for writing executor
/// testcases.
void M::registerGraphRTTestPrimitives(PrimitiveRegistry &registry) {
  registry.add("test.chain.new", test_chain_new);
  registry.add("test.print_hello", test_print_hello);
  registry.add("test.constant.i32", test_constant_i32);
  registry.add("test.print.i32", test_print_i32);
  registry.add("test.add.i32", test_add_i32);

  registry.add("test.fail", test_fail);
  registry.add("test.partial_fail", test_partial_fail);
}
