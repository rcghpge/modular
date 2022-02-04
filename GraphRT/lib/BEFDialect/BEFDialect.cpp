//===- BEFDialect.cpp - Implement the BEF dialect -------------------------===//
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

using namespace M;
using namespace BEF;

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
}
