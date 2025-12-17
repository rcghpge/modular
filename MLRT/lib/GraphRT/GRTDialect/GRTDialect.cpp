//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the GRT dialect.
//
//===----------------------------------------------------------------------===//

#include "MLRT/GraphRT/GRTDialect/GRTDialect.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace GRT;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "MLRT/GraphRT/GRTDialect/GRTDialect.cpp.inc"

void GRTDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "MLRT/GraphRT/GRTDialect/GRT.cpp.inc"
      >();
}
