//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MOPDialect/MOPAttrs.h"
#include "GenericML/MOPDialect/MOPDialect.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include "./MLIRFieldParsers.h"

using namespace M;
using namespace MOP;

//===----------------------------------------------------------------------===//
// BufferAttr
//===----------------------------------------------------------------------===//

/// Interns data into allocator with align (or 1 if no align constraint given).
/// The BufferAttr's getData will thus return a pointer respecting the
/// requested alignment.
static ArrayRef<uint8_t>
copyIntoBuffer(mlir::StorageUniquer::StorageAllocator &allocator,
               ArrayRef<uint8_t> data, Optional<int64_t> align) {
  auto *ptr = static_cast<uint8_t *>(
      allocator.allocate(data.size(), align ? *align : sizeof(uint8_t)));
  std::uninitialized_copy(data.begin(), data.end(), ptr);
  return {ptr, data.size()};
}

/// (Just to make clear we'll be converting from decoded hex strings directly
///  to vectors of uint8_t below.)
static_assert(sizeof(uint8_t) == sizeof(char));

/// Parses a hex string into data. The empty string denotes no data.
static ParseResult parseBufferData(AsmParser &p,
                                   FailureOr<SmallVector<uint8_t>> &data) {
  std::string encoded;
  auto res = p.parseString(&encoded);
  if (failed(res))
    return res;
  if (encoded.empty()) {
    data.emplace();
    return success();
  }
  StringRef encodedRef = encoded;
  std::string decoded;
  auto startLoc = p.getCurrentLocation();
  if (!encodedRef.consume_front("0x") || encodedRef.empty() ||
      (encodedRef.size() & 1) || !llvm::tryGetFromHex(encodedRef, decoded)) {
    return p.emitError(startLoc,
                       "invalid hex string for constant buffer literal");
  }
  data.emplace(decoded.size());
  memcpy(data->data(), decoded.data(), decoded.size());
  return success();
}

/// Print data as a hex string. The empty data array is printed as the empty
/// string.
static void printBufferData(AsmPrinter &p, ArrayRef<uint8_t> data) {
  if (data.empty()) {
    p << "\"\"";
    return;
  }
  p << "\"0x";
  StringRef decodedRef(reinterpret_cast<const char *>(data.data()),
                       data.size());
  p << llvm::toHex(decodedRef);
  p << "\"";
}

/// Verifies the attribute's align constraint is sensible.
LogicalResult BufferAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 Optional<int64_t> align,
                                 ::llvm::ArrayRef<uint8_t> data) {
  if (align && *align < 0)
    return emitError() << "buffer align cannot be negative";
  if (align && !llvm::isPowerOf2_64(*align))
    return emitError() << "buffer align must be a power of two";
  return success();
}

/// To support the ElementsAttr interface, returns the underlying data.
FailureOr<const uint8_t *>
BufferAttr::try_value_begin_impl(OverloadToken<uint8_t>) const {
  return getData().data();
}

/// To support the ElementsAttr interface, returns the implied buffer type.
ShapedType BufferAttr::getType() const {
  return BufferType::get(getContext(), static_cast<int64_t>(getData().size()),
                         getAlign());
}

//===----------------------------------------------------------------------===//
// MOPDialect attribute support
//===----------------------------------------------------------------------===//

void MOPDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "GenericML/MOPDialect/MOPAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GenericML/MOPDialect/MOPAttrs.cpp.inc"
