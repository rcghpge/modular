//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MODialect/MODialect.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "Support/MDialect/MDialect.h"

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// MODialect
//===----------------------------------------------------------------------===//

void MODialect::initialize() {
  registerAttributes();
  registerTypes();
  registerOperations();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GenericML/MODialect/MODialect.cpp.inc"
