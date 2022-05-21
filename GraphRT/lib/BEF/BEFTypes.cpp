//===- BEFTypes.cpp -------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEF/BEFTypes.h"
#include "llvm/ADT/SmallVector.h"
using namespace M;

/// Return the encoded size of this type in bytes.
size_t BEFType::getEncodedSizeInBytes() const {
  auto typeID = getKind();
  // The 'isComplex' bit doesn't affect the length.
  typeID &= ~BEF::TypeKind::mComplex;

  switch (typeID) {
  case BEF::TypeKind::kF16:
  case BEF::TypeKind::kBF16:
  case BEF::TypeKind::kF32:
  case BEF::TypeKind::kF64:
  case BEF::TypeKind::kF80:
  case BEF::TypeKind::kF128:
  case BEF::TypeKind::kBool:
    // Primitive types all have a single-byte encoding.
    return 1;
  case BEF::TypeKind::kFunction:
    return this->cast<BEFFunctionType>().getEncodedSizeInBytes();

  default:
    // All signed/unsigned integers are single-byte encoding.
    if (((typeID & BEF::TypeKind::mSigned) == BEF::TypeKind::mSigned) ||
        ((typeID & BEF::TypeKind::mUnsigned) == BEF::TypeKind::mUnsigned))
      return 1;
    // Unknown types don't have a known length, just don't advance.
    return 0;
  case BEF::TypeKind::kTensor:
  case BEF::TypeKind::kQuantized:
    // TODO!
    return 0;
  }
}

//===----------------------------------------------------------------------===//
// BEFFunctionType
//===----------------------------------------------------------------------===//

/// Decode the inputs and results of a BEFFunctionTypeAttr into the specified
/// vectors and return the encoded size of the function type.
static size_t decodeFunctionType(const uint8_t *pointer,
                                 SmallVectorImpl<BEFType> &inputs,
                                 SmallVectorImpl<BEFType> &results) {
  auto *dataPtr = pointer;
  assert(BEF::TypeKind(dataPtr[0]) == BEF::TypeKind::kFunction);
  uint8_t numInputs = dataPtr[1];
  uint8_t numResults = dataPtr[2];
  dataPtr += 3;
  while (numInputs--) {
    inputs.push_back(BEFType(dataPtr));
    dataPtr += inputs.back().getEncodedSizeInBytes();
  }
  while (numResults--) {
    results.push_back(BEFType(dataPtr));
    dataPtr += results.back().getEncodedSizeInBytes();
  }
  return dataPtr - pointer;
}

/// Return the encoded size of this type in the BEF file, in bytes.
size_t BEFFunctionType::getEncodedSizeInBytes() const {
  SmallVector<BEFType> ignored;
  // Use decodeFunctionType to determine the size of the function type.
  return decodeFunctionType(pointer, ignored, ignored);
}

void BEFFunctionType::decode(SmallVectorImpl<BEFType> &inputs,
                             SmallVectorImpl<BEFType> &results) const {
  (void)decodeFunctionType(pointer, inputs, results);
}
