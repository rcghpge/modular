//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MOP dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MOPDialect/MOPOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace M;
using namespace MOP;

//===----------------------------------------------------------------------===//
// ConstantBufferOp
//===----------------------------------------------------------------------===//

// Verifies the attribute and result types are consistent, and the attribute
// is supported.
LogicalResult BufferConstantOp::verify() {
  auto value = getValue();
  if (value.isSplat())
    return emitOpError("does not support splat attributes");
  auto shape = value.getType();
  if (!shape.hasRank() || shape.getShape().size() != 1)
    return emitOpError("only supports rank-1 attributes");
  auto elemType = value.getElementType();
  if (!elemType.isUnsignedInteger(8))
    return emitOpError("only supports attributes of uint8_t elements")
           << " (have " << elemType << ")";
  auto bufferType = getResult().getType();
  if (!bufferType.getSize())
    return emitOpError("requires the buffer type have a fixed size");
  if (*bufferType.getSize() != value.getNumElements())
    return emitOpError("requires the buffer type have a size matching the "
                       "attribute size")
           << " (" << *bufferType.getSize() << " vs " << value.getNumElements()
           << ")";
  auto typeAlign = bufferType.getAlign();

  // Only BufferAttr and DenseResourceElementsAttr can convey an align
  // constraint.
  if (auto bufferAttr = mlir::dyn_cast<BufferAttr>(value)) {
    auto bufferAlign = bufferAttr.getAlign();
    if (typeAlign.has_value() != bufferAlign.has_value())
      return emitOpError(
          "requires the buffer type have an align constraint if and only "
          "if the attribute has an align constraint");
    if (typeAlign.has_value() && *typeAlign != *bufferAlign)
      return emitOpError("requires the buffer type align match the "
                         "attribute align")
             << " (" << *typeAlign << " vs " << *bufferAlign << ")";
  } else if (auto resourceAttr =
                 mlir::dyn_cast<mlir::DenseResourceElementsAttr>(value)) {
    auto handle = resourceAttr.getRawHandle();
    auto blob = handle.getBlob();
    if (!blob)
      return emitOpError("requires the attribute resource to be initialized");
    if (blob->getData().size() != static_cast<size_t>(value.getNumElements()))
      // TODO: Why is this not enforced by the dense_resource attribute?
      return emitOpError("the attribute resource size does not match the "
                         "attribute elements size")
             << " (" << blob->getData().size() << " vs "
             << value.getNumElements() << ")";
    auto blobAlign = blob->getDataAlignment();
    if (typeAlign.has_value() != (blobAlign != 0))
      return emitOpError(
          "requires the buffer type have an align constraint if and only "
          "if the attribute has an align constraint");
    if (typeAlign.has_value() && static_cast<size_t>(*typeAlign) != blobAlign)
      return emitOpError("requires the buffer type align match the "
                         "attribute align")
             << " (" << *typeAlign << " vs " << blobAlign << ")";
  } else {
    if (typeAlign.has_value())
      return emitOpError("requires the buffer type not have an align since the "
                         "attribute does not convey any align constraint");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MOPDialect op support
//===----------------------------------------------------------------------===//

void MOPDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GenericML/MOPDialect/MOP.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/MOPDialect/MOP.cpp.inc"
