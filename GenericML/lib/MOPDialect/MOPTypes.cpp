//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MOPDialect/MOPTypes.h"
#include "GenericML/MOPDialect/MOPDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

#include "./MLIRFieldParsers.h"

using namespace M;
using namespace MOP;

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

// Verifies the buffer size and align constraints are valid.
mlir::LogicalResult
BufferType::verify(function_ref<InFlightDiagnostic()> emitError,
                   Optional<int64_t> size, Optional<int64_t> align) {
  if (size && *size < 0)
    return emitError() << "buffer size cannot be negative";
  if (align && *align < 0)
    return emitError() << "buffer align cannot be negative";
  if (align && !llvm::isPowerOf2_64(*align))
    return emitError() << "buffer align must be a power of two";
  return success();
}

// Implements ShapedType::cloneWith
ShapedType BufferType::cloneWith(llvm::Optional<llvm::ArrayRef<long>> shape,
                                 mlir::Type dtype) const {
  assert(!shape && "cannot change shape in BufferType::cloneWith");
  assert(dtype.isUnsignedInteger(8) &&
         "invalid dtype for BufferType::cloneWith");
  return BufferType::get(getContext(), getSize(), getAlign());
}

// Implements ShapedType::getElementType
Type BufferType::getElementType() const {
  return mlir::IntegerType::get(
      getContext(), /*width=*/8,
      mlir::IntegerType::SignednessSemantics::Unsigned);
}

// Implements ShapedType::hasRank
bool BufferType::hasRank() const { return true; }

// We defer BufferType::getShape till after ODS import since depends on impl.

//===----------------------------------------------------------------------===//
// MOPDialect type support
//===----------------------------------------------------------------------===//

void MOPDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "GenericML/MOPDialect/MOPTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "GenericML/MOPDialect/MOPTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// BufferType (cont.)
//===----------------------------------------------------------------------===//

const int64_t BufferType::unknownSize = mlir::ShapedType::kDynamicSize;

// Implements ShapedType::getShape
ArrayRef<int64_t> BufferType::getShape() const {
  Optional<int64_t> &size = getImpl()->size;
  if (size) {
    return ArrayRef<int64_t>(&size.value(), 1);
  } else {
    return ArrayRef<int64_t>(&unknownSize, 1);
  }
}
