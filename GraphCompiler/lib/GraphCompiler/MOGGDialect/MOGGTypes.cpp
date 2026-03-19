//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/GraphCompiler/MOSHDialect/MOSHTypes.h"

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MOGG;

//===----------------------------------------------------------------------===//
// MOGGDialect type support
//===----------------------------------------------------------------------===//

void MOGGDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGTypes.cpp.inc"
