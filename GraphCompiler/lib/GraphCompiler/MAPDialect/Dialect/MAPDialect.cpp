//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "GraphCompiler/GraphCompiler/MAPDialect/Dialect/MAPAttrs.h"
#include "GraphCompiler/GraphCompiler/MAPDialect/Dialect/MAPOps.h"

using namespace M;
using namespace MAP;

//===----------------------------------------------------------------------===//
// MAPDialect
//===----------------------------------------------------------------------===//

void MAPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "GraphCompiler/GraphCompiler/MAPDialect/Dialect/MAP.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "GraphCompiler/GraphCompiler/MAPDialect/Dialect/MAPAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MAPDialect/Dialect/MAPDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "GraphCompiler/GraphCompiler/MAPDialect/Dialect/MAPAttrs.cpp.inc"
