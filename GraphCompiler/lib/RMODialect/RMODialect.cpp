//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/RMODialect/RMODialect.h"
#include "GraphCompiler/MODialect/MODialect.h"

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

#include "GraphCompiler/RMODialect/RMODialect.cpp.inc"
