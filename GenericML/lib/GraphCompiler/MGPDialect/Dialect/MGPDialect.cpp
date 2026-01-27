//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPDialect.h"
#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPTensorInterface.h"
#include "MLRT/GraphRT/MEFDialect/MEFDialect.h"
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

#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPDialect.cpp.inc"
