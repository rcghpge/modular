//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MEF dialect.
//
//===----------------------------------------------------------------------===//

#include "MLRT/GraphRT/MEFDialect/MEFDialect.h"
#include "MLRT/GraphRT/MEFDialect/MEFOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MEF;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "MLRT/GraphRT/MEFDialect/MEFDialect.cpp.inc"

void MEFDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "MLRT/GraphRT/MEFDialect/MEF.cpp.inc"
      >();

  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "MLRT/GraphRT/MEFDialect/MEFTypes.cpp.inc"
      >();
}
