//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPAttrs.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MAP;

//===----------------------------------------------------------------------===//
// MAPDialect
//===----------------------------------------------------------------------===//

void MAPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "GraphCompiler/MAPDialect/Dialect/MAP.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "GraphCompiler/MAPDialect/Dialect/MAPTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "GraphCompiler/MAPDialect/Dialect/MAPAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "GraphCompiler/MAPDialect/Dialect/MAPTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "GraphCompiler/MAPDialect/Dialect/MAPAttrs.cpp.inc"
