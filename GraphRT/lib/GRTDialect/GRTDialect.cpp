//===- GRTDialect.cpp -----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the GRT dialect.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/GRTDialect/GRTDialect.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace GRT;

//===----------------------------------------------------------------------===//
// Dialect Type Parsing and Printing
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#define GET_TYPEDEF_CLASSES
#include "GraphRT/GRTDialect/GRTTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "GraphRT/GRTDialect/GRTDialect.cpp.inc"

void GRTDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "GraphRT/GRTDialect/GRT.cpp.inc"
      >();

  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "GraphRT/GRTDialect/GRTTypes.cpp.inc"
      >();
}
