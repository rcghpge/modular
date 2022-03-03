//===- BEFDialect.cpp -----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the BEF dialect.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFDialect/BEFDialect.h"
#include "GraphRT/BEFDialect/BEFOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace BEF;

//===----------------------------------------------------------------------===//
// Dialect Type Parsing and Printing
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#define GET_TYPEDEF_CLASSES
#include "GraphRT/BEFDialect/BEFTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "GraphRT/BEFDialect/BEFDialect.cpp.inc"

void BEFDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "GraphRT/BEFDialect/BEF.cpp.inc"
      >();

  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "GraphRT/BEFDialect/BEFTypes.cpp.inc"
      >();
}
