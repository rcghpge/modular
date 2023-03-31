//===----------------------------------------------------------------------===//
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

#include "GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "LLCL/Runtime/Runtime.h"
#include "llvm/ADT/StringRef.h"
using namespace M;

//===----------------------------------------------------------------------===//
// Integer ops
//===----------------------------------------------------------------------===//

static bool constant_i1(MEFAttr<bool> value) { return *value; }

static int32_t constant_si32(MEFAttr<int32_t> value) { return *value; }

static Chain print_si32(int32_t value, Chain chain) {
  llvm::outs() << "int32 = " << value << "\n";
  llvm::outs().flush();
  return chain;
}

static int32_t add_si32(int32_t lhs, int32_t rhs) { return lhs + rhs; }

static bool lessequal_si32(int32_t lhs, int32_t rhs) { return lhs <= rhs; }

/// This registers the 'graphrt' dialect integer primitives.
void M::registerGraphRTIntegerPrimitives(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "grt.constant.i1", constant_i1);
  ADD_PRIMITIVE(registry, "grt.constant.si32", constant_si32);
  ADD_PRIMITIVE(registry, "grt.add.si32", add_si32);
  ADD_PRIMITIVE(registry, "grt.lessequal.si32", lessequal_si32);
  ADD_PRIMITIVE(registry, "grt.print.si32", print_si32);
}
