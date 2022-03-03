//===- GraphRTDialect.cpp -------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the GraphRT dialect.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/GraphRTDialect/GraphRTDialect.h"
#include "GraphRT/GraphRTDialect/GraphRTOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace GraphRT;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "GraphRT/GraphRTDialect/GraphRTDialect.cpp.inc"

void GraphRTDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "GraphRT/GraphRTDialect/GraphRT.cpp.inc"
      >();

  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "GraphRT/GraphRTDialect/GraphRTTypes.cpp.inc"
      >();

  // We are flexible with operations that exist in the GraphRT namespace.
  allowUnknownOperations(true);
}
