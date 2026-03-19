//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the RMO dialect attributes.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/RMODialect/RMOAttrs.h"
#include "GraphCompiler/GraphCompiler/RMODialect/RMODialect.h"

using namespace M;
using namespace RMO;

//===----------------------------------------------------------------------===//
// RMODialect attribute support
//===----------------------------------------------------------------------===//

void RMODialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "GraphCompiler/GraphCompiler/RMODialect/RMOAttrs.cpp.inc"
      >();
  addAttributes<PythonFrameAttr>();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GraphCompiler/GraphCompiler/RMODialect/RMOAttrs.cpp.inc"
