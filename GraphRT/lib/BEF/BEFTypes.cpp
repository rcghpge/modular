//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEF/BEFTypes.h"
#include "Support/ML/TensorSpec.h"
#include "llvm/ADT/SmallVector.h"
using namespace M;

/// Return the encoded size of this type in bytes.
size_t BEFType::getEncodedSizeInBytes() const {
  // The 'isComplex' bit doesn't affect the length.
  auto typeID = getKind().stripComplex();

  switch (typeID.getValue()) {
  case BEF::TypeKind::f16:
  case BEF::TypeKind::bf16:
  case BEF::TypeKind::f32:
  case BEF::TypeKind::f64:
  case BEF::TypeKind::f80:
  case BEF::TypeKind::f128:
  case BEF::TypeKind::kBool:
    // Primitive types all have a single-byte encoding.
    return 1;
  case BEF::TypeKind::function:
    return cast<BEFFunctionType>(this)->getEncodedSizeInBytes();
  case BEF::TypeKind::tensor:
    return cast<BEFTensorType>(this)->getEncodedSizeInBytes();
  case BEF::TypeKind::quantized:
    return cast<BEFQuantizedType>(this)->getEncodedSizeInBytes();
  case BEF::TypeKind::quantizedPerAxis:
    return cast<BEFQuantizedPerAxisType>(this)->getEncodedSizeInBytes();
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
static size_t decodeFunctionType(const uint8_t *pointer, const uint8_t *attrEnd,
                                 SmallVectorImpl<BEFType> &inputs,
                                 SmallVectorImpl<BEFType> &results) {
  auto *dataPtr = pointer;
  assert(pointer <= attrEnd &&
         "BEF datum not within bounds of the BEF Attribute Section");
  assert(BEF::TypeKind(dataPtr[0]) == BEF::TypeKind::function);
  uint8_t numInputs = dataPtr[1];
  uint8_t numResults = dataPtr[2];
  dataPtr += 3;
  while (numInputs--) {
    uint8_t attrSize = BEFType(dataPtr, attrEnd).getEncodedSizeInBytes();
    inputs.push_back(BEFType(dataPtr, dataPtr + attrSize));
    dataPtr += attrSize;
  }
  while (numResults--) {
    uint8_t attrSize = BEFType(dataPtr, attrEnd).getEncodedSizeInBytes();
    results.push_back(BEFType(dataPtr, dataPtr + attrSize));
    dataPtr += attrSize;
  }
  return dataPtr - pointer;
}

/// Return the encoded size of this type in the BEF file, in bytes.
size_t BEFFunctionType::getEncodedSizeInBytes() const {
  SmallVector<BEFType> ignored;
  // Use decodeFunctionType to determine the size of the function type.
  return decodeFunctionType(pointer, attrEnd, ignored, ignored);
}

void BEFFunctionType::decode(SmallVectorImpl<BEFType> &inputs,
                             SmallVectorImpl<BEFType> &results) const {
  (void)decodeFunctionType(pointer, attrEnd, inputs, results);
}

//===----------------------------------------------------------------------===//
// BEFTensorType
//===----------------------------------------------------------------------===//

/// Decode the inputs and results of a BEFTensorType into the specified
/// vector and return the encoded size of the type.
static size_t decodeTensorType(const uint8_t *pointer, const uint8_t *attrEnd,
                               BEFType &elementType,
                               SmallVectorImpl<int64_t> &dims) {
  auto *dataPtr = pointer;
  assert(pointer <= attrEnd &&
         "BEF datum not within bounds of the BEF Attribute Section");
  assert(BEF::TypeKind(dataPtr[0]) == BEF::TypeKind::tensor);
  ++dataPtr;
  uint8_t attrSize = BEFType(dataPtr, attrEnd).getEncodedSizeInBytes();
  elementType = BEFType(dataPtr, pointer + attrSize);
  dataPtr += attrSize;
  uint8_t rank = *dataPtr++;

  while (rank--)
    dims.push_back(BEF::parseVBR(dataPtr));
  return dataPtr - pointer;
}

BEFType BEFTensorType::decode(SmallVectorImpl<int64_t> &dims) const {
  BEFType elementType(nullptr, nullptr);
  (void)decodeTensorType(pointer, attrEnd, elementType, dims);
  return elementType;
}

/// Return the encoded size of this type in the BEF file, in bytes.
size_t BEFTensorType::getEncodedSizeInBytes() const {
  BEFType elementType(nullptr, nullptr);
  SmallVector<int64_t> dims;
  return decodeTensorType(pointer, attrEnd, elementType, dims);
}

//===----------------------------------------------------------------------===//
// BEFQuantizedType
//===----------------------------------------------------------------------===//

// Helper to decode storage and expressed types when decoding quantized types.
static void decodeStorageAndExpressedType(const uint8_t *&dataPtr,
                                          const uint8_t *attrEnd,
                                          BEFType &storageType,
                                          BEFType &expressedType) {
  uint8_t storageAttrSize = BEFType(dataPtr, attrEnd).getEncodedSizeInBytes();
  storageType = BEFType(dataPtr, dataPtr + storageAttrSize);
  dataPtr += storageAttrSize;
  uint8_t expressedAttrSize = BEFType(dataPtr, attrEnd).getEncodedSizeInBytes();
  expressedType = BEFType(dataPtr, dataPtr + expressedAttrSize);
  dataPtr += expressedAttrSize;
}

/// Decode the inputs and results of a BEFQuantizedType into the specified
/// arguments and return the encoded size of the type.
static size_t decodeQuantizedType(const uint8_t *pointer,
                                  const uint8_t *attrEnd, BEFType &storageType,
                                  BEFType &expressedType, float &scale,
                                  int64_t &zeroPoint) {
  auto *dataPtr = pointer;
  assert(pointer <= attrEnd &&
         "BEF datum not within bounds of the BEF Attribute Section");
  assert(BEF::TypeKind(dataPtr[0]) == BEF::TypeKind::quantized);
  decodeStorageAndExpressedType(++dataPtr, attrEnd, storageType, expressedType);

  scale = BEF::parseFloat(dataPtr);
  zeroPoint = BEF::parseSignedVBR(dataPtr);

  return dataPtr - pointer;
}

std::pair<BEFType, BEFType> BEFQuantizedType::decode(float &scale,
                                                     int64_t &zeroPoint) const {
  BEFType empty(nullptr, nullptr);
  std::pair<BEFType, BEFType> storageAndExpressedTypes{empty, empty};
  (void)decodeQuantizedType(pointer, attrEnd, storageAndExpressedTypes.first,
                            storageAndExpressedTypes.second, scale, zeroPoint);
  return storageAndExpressedTypes;
}

/// Return the encoded size of this type in the BEF file, in bytes.
size_t BEFQuantizedType::getEncodedSizeInBytes() const {
  BEFType ignored(nullptr, nullptr);
  float scale;
  int64_t zeroPoint;
  return decodeQuantizedType(pointer, attrEnd, ignored, ignored, scale,
                             zeroPoint);
}

//===----------------------------------------------------------------------===//
// BEFQuantizedPerAxisType
//===----------------------------------------------------------------------===//

/// Decode the inputs and results of a BEFQuantizedPerAxisType into the
/// specified arguments and return the encoded size of the type.
static size_t decodeQuantizedPerAxisType(
    const uint8_t *pointer, const uint8_t *attrEnd, BEFType &storageType,
    BEFType &expressedType, SmallVectorImpl<float> &scales,
    SmallVectorImpl<int64_t> &zeroPoints, uint8_t &quantizedDim) {
  auto *dataPtr = pointer;
  assert(pointer <= attrEnd &&
         "BEF datum not within bounds of the BEF Attribute Section");
  assert(BEF::TypeKind(dataPtr[0]) == BEF::TypeKind::quantizedPerAxis);
  decodeStorageAndExpressedType(++dataPtr, attrEnd, storageType, expressedType);

  quantizedDim = *dataPtr++;
  auto numParams = BEF::parseVBR(dataPtr);

  // Decode the scales, then the zero points, and push them into their vectors.
  // Note that these two loops cannot be merged, because dataPtr is incremented
  // every time we decode an element.
  for (size_t j = 0; j < numParams; j++)
    scales.push_back(BEF::parseFloat(dataPtr));
  for (size_t j = 0; j < numParams; j++)
    zeroPoints.push_back(BEF::parseSignedVBR(dataPtr));

  return dataPtr - pointer;
}

std::pair<BEFType, BEFType>
BEFQuantizedPerAxisType::decode(SmallVectorImpl<float> &scales,
                                SmallVectorImpl<int64_t> &zeroPoints,
                                uint8_t &quantizedDim) const {
  BEFType empty(nullptr, nullptr);
  std::pair<BEFType, BEFType> storageAndExpressedTypes{empty, empty};
  (void)decodeQuantizedPerAxisType(
      pointer, attrEnd, storageAndExpressedTypes.first,
      storageAndExpressedTypes.second, scales, zeroPoints, quantizedDim);
  return storageAndExpressedTypes;
}

/// Return the encoded size of this type in the BEF file, in bytes.
size_t BEFQuantizedPerAxisType::getEncodedSizeInBytes() const {
  BEFType ignored(nullptr, nullptr);
  SmallVector<float> scales;
  SmallVector<int64_t> zeroPoints;
  uint8_t quantizedDim;
  return decodeQuantizedPerAxisType(pointer, attrEnd, ignored, ignored, scales,
                                    zeroPoints, quantizedDim);
}
