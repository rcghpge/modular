//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPDialect.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GraphRT/MEFDialect/MEFDialect.h"
#include "GraphRT/MIPDialect/MIPDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

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

Operation *MGPDialect::materializeConstant(OpBuilder &b, Attribute value,
                                           Type type, Location loc) {
  if (auto elements = dyn_cast_or_null<ElementsAttr>(value))
    return b.create<MGP::CpuBufferConstantOp>(loc, type, elements);

  return nullptr;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPDialect.cpp.inc"
