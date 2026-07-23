//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHOps.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"

using namespace M;
using namespace MOSH;

//===----------------------------------------------------------------------===//
// MOSHDialect
//===----------------------------------------------------------------------===//

void MOSHDialect::initialize() {
  registerAttributes();
  registerTypes();
  registerOperations();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOSHDialect/MOSHDialect.cpp.inc"
