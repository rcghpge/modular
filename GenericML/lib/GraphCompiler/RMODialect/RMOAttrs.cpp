//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the RMO dialect attributes.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/RMODialect/RMOAttrs.h"
#include "GenericML/GraphCompiler/RMODialect/RMODialect.h"

using namespace M;
using namespace RMO;

//===----------------------------------------------------------------------===//
// RMODialect attribute support
//===----------------------------------------------------------------------===//

void RMODialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "GenericML/GraphCompiler/RMODialect/RMOAttrs.cpp.inc"
      >();
  addAttributes<PythonFrameAttr>();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GenericML/GraphCompiler/RMODialect/RMOAttrs.cpp.inc"
