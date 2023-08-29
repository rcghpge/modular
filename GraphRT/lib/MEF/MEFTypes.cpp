//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/MEF/MEFTypes.h"
#include "Support/ML/TensorSpec.h"
#include "llvm/ADT/SmallVector.h"
using namespace M;

/// Return the encoded size of this type in bytes.
size_t MEFType::getEncodedSizeInBytes() const {
  // The 'isComplex' bit doesn't affect the length.
  auto typeID = getKind().stripComplex();

  switch (typeID.getValue()) {
  case MEF::TypeKind::f16:
  case MEF::TypeKind::bf16:
  case MEF::TypeKind::f32:
  case MEF::TypeKind::f64:
  case MEF::TypeKind::f80:
  case MEF::TypeKind::f128:
  case MEF::TypeKind::kBool:
    // Primitive types all have a single-byte encoding.
    return 1;
  case MEF::TypeKind::function:
    return cast<MEFFunctionType>(this)->getEncodedSizeInBytes();
  case MEF::TypeKind::tensor:
    return cast<MEFTensorType>(this)->getEncodedSizeInBytes();
  case MEF::TypeKind::quantized:
    return cast<MEFQuantizedType>(this)->getEncodedSizeInBytes();
  case MEF::TypeKind::quantizedPerAxis:
    return cast<MEFQuantizedPerAxisType>(this)->getEncodedSizeInBytes();
  default:
    // All signed/unsigned integers are single-byte encoding.
    if (typeID.isInt())
      return 1;
    // Unknown types don't have a known length, just don't advance.
    return 0;
  }
}

//===----------------------------------------------------------------------===//
// MEFFunctionType
//===----------------------------------------------------------------------===//

/// Decode the inputs and results of a MEFFunctionType into the specified
/// vectors and return the encoded size of the function type.
static size_t decodeFunctionType(const uint8_t *pointer, const uint8_t *attrEnd,
                                 SmallVectorImpl<MEFType> &inputs,
                                 SmallVectorImpl<MEFType> &results) {
  auto *dataPtr = pointer;
  assert(pointer <= attrEnd &&
         "MEF datum not within bounds of the MEF Attribute Section");
  assert(MEF::TypeKind(dataPtr[0]) == MEF::TypeKind::function);
  uint8_t numInputs = dataPtr[1];
  uint8_t numResults = dataPtr[2];
  dataPtr += 3;
  while (numInputs--) {
    size_t attrSize = MEFType(dataPtr, attrEnd).getEncodedSizeInBytes();
    inputs.push_back(MEFType(dataPtr, dataPtr + attrSize));
    dataPtr += attrSize;
  }
  while (numResults--) {
    size_t attrSize = MEFType(dataPtr, attrEnd).getEncodedSizeInBytes();
    results.push_back(MEFType(dataPtr, dataPtr + attrSize));
    dataPtr += attrSize;
  }
  return dataPtr - pointer;
}

/// Return the encoded size of this type in the MEF file, in bytes.
size_t MEFFunctionType::getEncodedSizeInBytes() const {
  SmallVector<MEFType> ignored;
  // Use decodeFunctionType to determine the size of the function type.
  return decodeFunctionType(pointer, attrEnd, ignored, ignored);
}

void MEFFunctionType::decode(SmallVectorImpl<MEFType> &inputs,
                             SmallVectorImpl<MEFType> &results) const {
  (void)decodeFunctionType(pointer, attrEnd, inputs, results);
}

//===----------------------------------------------------------------------===//
// MEFTensorType
//===----------------------------------------------------------------------===//

/// Decode the inputs and results of a MEFTensorType into the specified
/// vector and return the encoded size of the type.
static size_t decodeTensorType(const uint8_t *pointer, const uint8_t *attrEnd,
                               MEFType &elementType, bool &hasRank,
                               SmallVectorImpl<int64_t> &dims) {
  auto *dataPtr = pointer;
  assert(pointer <= attrEnd &&
         "MEF datum not within bounds of the MEF Attribute Section");
  // Tensor type tag.
  assert(MEF::TypeKind(dataPtr[0]) == MEF::TypeKind::tensor);
  ++dataPtr;
  // Element type.
  size_t attrSize = MEFType(dataPtr, attrEnd).getEncodedSizeInBytes();
  elementType = MEFType(dataPtr, pointer + attrSize);
  dataPtr += attrSize;
  // Rank.
  uint8_t rank = *dataPtr++;
  if (rank == Detail::kDynamicRank) {
    // Unranked
    hasRank = false;
  } else {
    assert(rank <= kMaxRank);
    hasRank = true;
    // Dimensions.
    while (rank--)
      dims.push_back(MEF::parseVBR(dataPtr));
  }
  return dataPtr - pointer;
}

MEFType MEFTensorType::decodeStatic(SmallVectorImpl<int64_t> &dims) const {
  bool hasRank;
  MEFType elementType = decode(hasRank, dims);
  assert(hasRank && "concrete tensor types must have statically known ranks");
  assert(llvm::all_of(dims, [](int64_t dim) { return dim >= 0; }) &&
         "concrete tensor types must have statically known dimensions");
  return elementType;
}

MEFType MEFTensorType::decode(bool &hasRank,
                              SmallVectorImpl<int64_t> &dims) const {
  MEFType elementType(nullptr, nullptr);
  (void)decodeTensorType(pointer, attrEnd, elementType, hasRank, dims);
  return elementType;
}

/// Return the encoded size of this type in the MEF file, in bytes.
size_t MEFTensorType::getEncodedSizeInBytes() const {
  MEFType elementType(nullptr, nullptr);
  bool hasRank;
  SmallVector<int64_t> dims;
  return decodeTensorType(pointer, attrEnd, elementType, hasRank, dims);
}

//===----------------------------------------------------------------------===//
// MEFQuantizedType
//===----------------------------------------------------------------------===//

// Helper to decode storage and expressed types when decoding quantized types.
static void decodeStorageAndExpressedType(const uint8_t *&dataPtr,
                                          const uint8_t *attrEnd,
                                          MEFType &storageType,
                                          MEFType &expressedType) {
  size_t storageAttrSize = MEFType(dataPtr, attrEnd).getEncodedSizeInBytes();
  storageType = MEFType(dataPtr, dataPtr + storageAttrSize);
  dataPtr += storageAttrSize;
  size_t expressedAttrSize = MEFType(dataPtr, attrEnd).getEncodedSizeInBytes();
  expressedType = MEFType(dataPtr, dataPtr + expressedAttrSize);
  dataPtr += expressedAttrSize;
}

/// Decode the inputs and results of a MEFQuantizedType into the specified
/// arguments and return the encoded size of the type.
static size_t decodeQuantizedType(const uint8_t *pointer,
                                  const uint8_t *attrEnd, MEFType &storageType,
                                  MEFType &expressedType, float &scale,
                                  int64_t &zeroPoint) {
  auto *dataPtr = pointer;
  assert(pointer <= attrEnd &&
         "MEF datum not within bounds of the MEF Attribute Section");
  assert(MEF::TypeKind(dataPtr[0]) == MEF::TypeKind::quantized);
  decodeStorageAndExpressedType(++dataPtr, attrEnd, storageType, expressedType);

  scale = MEF::parseFloat(dataPtr);
  zeroPoint = MEF::parseSignedVBR(dataPtr);

  return dataPtr - pointer;
}

std::pair<MEFType, MEFType> MEFQuantizedType::decode(float &scale,
                                                     int64_t &zeroPoint) const {
  MEFType empty(nullptr, nullptr);
  std::pair<MEFType, MEFType> storageAndExpressedTypes{empty, empty};
  (void)decodeQuantizedType(pointer, attrEnd, storageAndExpressedTypes.first,
                            storageAndExpressedTypes.second, scale, zeroPoint);
  return storageAndExpressedTypes;
}

/// Return the encoded size of this type in the MEF file, in bytes.
size_t MEFQuantizedType::getEncodedSizeInBytes() const {
  MEFType ignored(nullptr, nullptr);
  float scale;
  int64_t zeroPoint;
  return decodeQuantizedType(pointer, attrEnd, ignored, ignored, scale,
                             zeroPoint);
}

//===----------------------------------------------------------------------===//
// MEFQuantizedPerAxisType
//===----------------------------------------------------------------------===//

/// Decode the inputs and results of a MEFQuantizedPerAxisType into the
/// specified arguments and return the encoded size of the type.
static size_t decodeQuantizedPerAxisType(
    const uint8_t *pointer, const uint8_t *attrEnd, MEFType &storageType,
    MEFType &expressedType, SmallVectorImpl<float> &scales,
    SmallVectorImpl<int64_t> &zeroPoints, uint8_t &quantizedDim) {
  auto *dataPtr = pointer;
  assert(pointer <= attrEnd &&
         "MEF datum not within bounds of the MEF Attribute Section");
  assert(MEF::TypeKind(dataPtr[0]) == MEF::TypeKind::quantizedPerAxis);
  decodeStorageAndExpressedType(++dataPtr, attrEnd, storageType, expressedType);

  quantizedDim = *dataPtr++;
  auto numParams = MEF::parseVBR(dataPtr);

  // Decode the scales, then the zero points, and push them into their vectors.
  // Note that these two loops cannot be merged, because dataPtr is incremented
  // every time we decode an element.
  for (size_t j = 0; j < numParams; j++)
    scales.push_back(MEF::parseFloat(dataPtr));
  for (size_t j = 0; j < numParams; j++)
    zeroPoints.push_back(MEF::parseSignedVBR(dataPtr));

  return dataPtr - pointer;
}

std::pair<MEFType, MEFType>
MEFQuantizedPerAxisType::decode(SmallVectorImpl<float> &scales,
                                SmallVectorImpl<int64_t> &zeroPoints,
                                uint8_t &quantizedDim) const {
  MEFType empty(nullptr, nullptr);
  std::pair<MEFType, MEFType> storageAndExpressedTypes{empty, empty};
  (void)decodeQuantizedPerAxisType(
      pointer, attrEnd, storageAndExpressedTypes.first,
      storageAndExpressedTypes.second, scales, zeroPoints, quantizedDim);
  return storageAndExpressedTypes;
}

/// Return the encoded size of this type in the MEF file, in bytes.
size_t MEFQuantizedPerAxisType::getEncodedSizeInBytes() const {
  MEFType ignored(nullptr, nullptr);
  SmallVector<float> scales;
  SmallVector<int64_t> zeroPoints;
  uint8_t quantizedDim;
  return decodeQuantizedPerAxisType(pointer, attrEnd, ignored, ignored, scales,
                                    zeroPoints, quantizedDim);
}
