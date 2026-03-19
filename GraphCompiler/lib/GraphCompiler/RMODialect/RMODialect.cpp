//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/RMODialect/RMODialect.h"
#include "GraphCompiler/GraphCompiler/MODialect/MODialect.h"

using namespace M;
using namespace MO;
using namespace RMO;

//===----------------------------------------------------------------------===//
// RMODialect
//===----------------------------------------------------------------------===//

void RMODialect::initialize() {
  registerAttributes();
  registerOperations();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/RMODialect/RMODialect.cpp.inc"
