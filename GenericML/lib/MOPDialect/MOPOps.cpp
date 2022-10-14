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
// BufferConstantOp
//===----------------------------------------------------------------------===//

/// Returns the align constraint for the given attribute, or none if it
/// has no such constraint.
static FailureOr<Optional<int64_t>>
getImpliedAlignment(ElementsAttr value,
                    function_ref<InFlightDiagnostic()> emitError) {
  if (auto bufferAttr = dyn_cast<BufferAttr>(value))
    return bufferAttr.getAlign();

  if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(value)) {
    auto handle = resourceAttr.getRawHandle();
    auto blob = handle.getBlob();
    if (!blob)
      return emitError() << "requires the attribute resource to be initialized";
    if (blob->getData().size() != static_cast<size_t>(value.getNumElements()))
      // TODO: Why is this not enforced by the dense_resource attribute?
      return emitError() << "the attribute resource size does not match the "
                            "attribute elements size"
                         << " (" << blob->getData().size() << " vs "
                         << value.getNumElements() << ")";
    auto blobAlign = blob->getDataAlignment();
    if (blobAlign == 0)
      // No align constraint.
      return {{}};
    // We have an align constraint.
    return Optional<int64_t>(static_cast<int64_t>(blobAlign));
  }

  // No align constraint.
  return {{}};
}

/// Verifies the attribute and result types are consistent, and the attribute
/// is supported.
static LogicalResult
verifyBufferConstantOp(BufferConstantOp &op,
                       function_ref<InFlightDiagnostic()> emitError) {
  // We can only construct from attributes which are dense arrays (ie rank-1)
  // of uint8_ts.
  auto value = op.getValue();
  if (value.isSplat())
    return emitError() << "does not support splat attributes";
  auto shape = value.getType();
  if (!shape.hasRank() || shape.getShape().size() != 1)
    return emitError() << "only supports rank-1 attributes";
  auto elemType = value.getElementType();
  if (!elemType.isUnsignedInteger(8))
    return emitError() << "only supports attributes of uint8_t elements"
                       << " (have " << elemType << ")";

  // The resulting constant's type must be sized, and the size must match
  // the attribute's size.
  auto bufferType = op.getResult().getType();
  if (!bufferType.getSize())
    return emitError() << "requires the buffer type have a fixed size";
  if (*bufferType.getSize() != value.getNumElements())
    return emitError() << "requires the buffer type have a size matching the "
                          "attribute size"
                       << " (" << *bufferType.getSize() << " vs "
                       << value.getNumElements() << ")";

  // Only BufferAttr and DenseResourceElementsAttr can convey an align
  // constraint.
  auto typeAlign = bufferType.getAlign();
  auto failOrValueAlign = getImpliedAlignment(value, emitError);
  if (failed(failOrValueAlign))
    return failOrValueAlign;
  auto valueAlign = *failOrValueAlign;
  if (typeAlign.has_value() != valueAlign.has_value())
    return emitError() << "requires the buffer type have an align "
                          "constraint if and only "
                          "if the attribute has an align constraint";
  if (typeAlign.has_value() && *typeAlign != *valueAlign)
    return emitError() << "requires the buffer type align match the "
                          "attribute align"
                       << " (" << *typeAlign << " vs " << *valueAlign << ")";

  return success();
}

/// Verifies the attribute and result types are consistent, and the attribute
/// is supported.
LogicalResult BufferConstantOp::verify() {
  return verifyBufferConstantOp(*this,
                                [this]() { return this->emitOpError(); });
}

//===----------------------------------------------------------------------===//
// ContextNewOp
//===----------------------------------------------------------------------===//

// Nothing to implement.

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
