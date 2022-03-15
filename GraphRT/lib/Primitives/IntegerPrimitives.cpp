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

static bool constant_i1(BEFAttribute<int8_t> value) { return *value != 0; }

static int32_t constant_i32(BEFAttribute<int32_t> value) { return *value; }

static Chain print_i32(int32_t value, Chain chain) {
  printf("int32 = %d\n", value);
  fflush(stdout);
  return chain;
}

static int32_t add_i32(int32_t lhs, int32_t rhs) { return lhs + rhs; }

static bool lessequal_i32(int32_t lhs, int32_t rhs) { return lhs <= rhs; }

/// This registers the 'graphrt' dialect integer primitives.
void M::registerGraphRTIntegerPrimitives(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "grt.constant.i1", constant_i1);
  ADD_PRIMITIVE(registry, "grt.constant.i32", constant_i32);
  ADD_PRIMITIVE(registry, "grt.add.i32", add_i32);
  ADD_PRIMITIVE(registry, "grt.lessequal.i32", lessequal_i32);
  ADD_PRIMITIVE(registry, "grt.print.i32", print_i32);
}
