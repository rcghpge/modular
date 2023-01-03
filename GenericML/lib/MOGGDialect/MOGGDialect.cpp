//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MOGGDialect/MOGGDialect.h"
#include "GenericML/MODialect/MODialect.h"
#include "Support/MDialect/MDialect.h"

using namespace M;
using namespace MOGG;

//===----------------------------------------------------------------------===//
// MOGGDialect
//===----------------------------------------------------------------------===//

void MOGGDialect::initialize() { registerOperations(); }

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GenericML/MOGGDialect/MOGGDialect.cpp.inc"
