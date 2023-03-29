//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPDialect.h"
#include "GraphRT/MEFDialect/MEFDialect.h"
#include "GraphRT/MIPDialect/MIPDialect.h"

using namespace M;
using namespace MGP;

//===----------------------------------------------------------------------===//
// MGPDialect
//===----------------------------------------------------------------------===//

void MGPDialect::initialize() {
  registerTypes();
  registerAttributes();
  registerOperations();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPDialect.cpp.inc"
