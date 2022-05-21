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
  // The 'isComplex' bit doesn't affect the length.
  auto typeID = getKind().stripComplex();

  switch (typeID.getValue()) {
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
  case BEF::TypeKind::kTensor:
    return this->cast<BEFTensorType>().getEncodedSizeInBytes();
  case BEF::TypeKind::kQuantized:
    return this->cast<BEFQuantizedType>().getEncodedSizeInBytes();
  default:
    // All signed/unsigned integers are single-byte encoding.
    if (typeID.isInt())
      return 1;
    // Unknown types don't have a known length, just don't advance.
    return 0;
  }
}

//===----------------------------------------------------------------------===//
// BEFFunctionType
//===----------------------------------------------------------------------===//

/// Decode the inputs and results of a BEFFunctionType into the specified
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

//===----------------------------------------------------------------------===//
// BEFTensorType
//===----------------------------------------------------------------------===//

/// Decode the inputs and results of a BEFTensorType into the specified
/// vector and return the encoded size of the type.
static size_t decodeTensorType(const uint8_t *pointer, BEFType &elementType,
                               SmallVectorImpl<int64_t> &dims) {
  auto *dataPtr = pointer;
  assert(BEF::TypeKind(dataPtr[0]) == BEF::TypeKind::kTensor);
  elementType = BEFType(++dataPtr);
  dataPtr += elementType.getEncodedSizeInBytes();
  uint8_t rank = *dataPtr++;

  while (rank--)
    dims.push_back(BEF::parseVBR(dataPtr));
  return dataPtr - pointer;
}

BEFType BEFTensorType::decode(SmallVectorImpl<int64_t> &dims) const {
  BEFType elementType(nullptr);
  (void)decodeTensorType(pointer, elementType, dims);
  return elementType;
}

/// Return the encoded size of this type in the BEF file, in bytes.
size_t BEFTensorType::getEncodedSizeInBytes() const {
  BEFType elementType(nullptr);
  SmallVector<int64_t> dims;
  return decodeTensorType(pointer, elementType, dims);
}

//===----------------------------------------------------------------------===//
// BEFQuantizedType
//===----------------------------------------------------------------------===//

/// Decode the inputs and results of a BEFQuantizedType into the specified
/// arguments and return the encoded size of the type.
static size_t decodeQuantizedType(const uint8_t *pointer, BEFType &storageType,
                                  BEFType &expressedType, float &scale,
                                  int64_t &zeroPoint) {
  auto *dataPtr = pointer;
  assert(BEF::TypeKind(dataPtr[0]) == BEF::TypeKind::kQuantized);
  storageType = BEFType(++dataPtr);
  dataPtr += storageType.getEncodedSizeInBytes();
  expressedType = BEFType(dataPtr);
  dataPtr += expressedType.getEncodedSizeInBytes();

  scale = *(const float *)dataPtr;
  dataPtr += sizeof(float);
  zeroPoint = BEF::parseSignedVBR(dataPtr);
  return dataPtr - pointer;
}

std::pair<BEFType, BEFType> BEFQuantizedType::decode(float &scale,
                                                     int64_t &zeroPoint) const {
  std::pair<BEFType, BEFType> storageExpressedTypes{nullptr, nullptr};
  (void)decodeQuantizedType(pointer, storageExpressedTypes.first,
                            storageExpressedTypes.second, scale, zeroPoint);
  return storageExpressedTypes;
}

/// Return the encoded size of this type in the BEF file, in bytes.
size_t BEFQuantizedType::getEncodedSizeInBytes() const {
  BEFType ignored(nullptr);
  float scale;
  int64_t zeroPoint;
  return decodeQuantizedType(pointer, ignored, ignored, scale, zeroPoint);
}