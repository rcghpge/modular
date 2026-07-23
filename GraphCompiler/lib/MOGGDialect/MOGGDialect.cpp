//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "Support/MDialect/MDialect.h"

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
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
// MOGGDialect
//===----------------------------------------------------------------------===//

void MOGGDialect::initialize() {
  registerAttributes();
  registerTypes();
  registerOperations();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/MOGGDialect.cpp.inc"
