//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/RMODialect/RMODialect.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"

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

#include "GenericML/GraphCompiler/RMODialect/RMODialect.cpp.inc"
