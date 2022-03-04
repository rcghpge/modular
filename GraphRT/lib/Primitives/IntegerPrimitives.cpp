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

#include "GraphRT/Primitives.h"

#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "LLCL/Runtime/Runtime.h"
#include "llvm/ADT/StringRef.h"
using namespace M;

//===----------------------------------------------------------------------===//
// Integer ops
//===----------------------------------------------------------------------===//

static void graphrt_constant_i32(PrimitiveFrame &frame) {
  assert(frame.getNumArgs() == 0 && frame.getNumResults() == 1 &&
         frame.getNumAttrs() == 1);
  frame.emplaceResult<int32_t>(0, frame.getAttr<int32_t>(0));
}

static Chain graphrt_print_i32(int32_t value, Chain chain) {
  printf("int32 = %d\n", value);
  fflush(stdout);
  return chain;
}

static int32_t graphrt_add_i32(int32_t lhs, int32_t rhs) { return lhs + rhs; }

/// This registers the 'graphrt' dialect integer primitives.
void M::registerGraphRTIntegerPrimitives(PrimitiveRegistry &registry) {
  registry.add("grt.constant.i32", graphrt_constant_i32);
  registry.add("grt.add.i32", M_PRIMITIVE(graphrt_add_i32));
  registry.add("grt.print.i32", M_PRIMITIVE(graphrt_print_i32));
}
