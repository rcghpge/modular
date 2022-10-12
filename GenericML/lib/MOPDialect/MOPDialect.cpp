//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MOPDialect/MOPDialect.h"

using namespace M;
using namespace MOP;

//===----------------------------------------------------------------------===//
// MOPDialect
//===----------------------------------------------------------------------===//

void MOPDialect::initialize() {
  registerTypes();
  registerAttributes();
  registerOperations();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GenericML/MOPDialect/MOPDialect.cpp.inc"
