//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPAttrs.h"
#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPDialect.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

using namespace M;
using namespace MGP;

//===----------------------------------------------------------------------===//
// MGPDialect attribute support
//===----------------------------------------------------------------------===//

void MGPDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPAttrs.cpp.inc"
