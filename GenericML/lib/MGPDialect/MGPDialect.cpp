//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPDialect.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTensorInterface.h"
#include "GraphRT/MEFDialect/MEFDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
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
  addInterface<MGPTensorBufferDialectInterface>();

  // The PrepareDevicePrimitivesPass rewrites device-generic ops to
  // include the device label in their name. We don't want to have
  // to register those since it's just for the convenience of the GraphRT
  // primitive dispatch mechanism.
  allowUnknownOperations();
}

Operation *MGPDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  if (auto intAttr = dyn_cast<mlir::IntegerAttr>(value))
    return mlir::index::ConstantOp::create(builder, loc, type, intAttr);

  return nullptr;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPDialect.cpp.inc"
