//===- TFLInputGeneration.cpp ---------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "TFLite/Tools/TFLInputGeneration.h"
#include "GenericML/Support/TensorDType.h"
#include "Support/Error.h"
#include "TFLite/Support/TFLiteType.h"
#include "TFLite/TFLRuntimeModel.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <random>

using namespace M;

template <typename T>
static LogicalResult doFill(InputGenerationKind inputGenKind, T *ptr,
                            size_t numElements) {
  bool isNegOnes = inputGenKind == InputGenerationKind::kNegOnes;
  if (std::is_integral_v<T> && std::is_unsigned_v<T> && isNegOnes)
    return failure();

  if (isNegOnes || inputGenKind == InputGenerationKind::kOnes) {
    std::fill(ptr, ptr + numElements, isNegOnes ? -1 : 1);
    return success();
  }

  if (inputGenKind == InputGenerationKind::kZeros) {
    memset(ptr, 0, numElements * sizeof(T));
    return success();
  }

  if (inputGenKind == InputGenerationKind::kRandom) {
    static std::default_random_engine randEngine(/*seed=*/0);
    auto fillRandom = [&](auto distr) {
      std::generate(ptr, ptr + numElements,
                    [&]() { return distr(randEngine); });
    };

    static_assert(std::is_arithmetic_v<T>,
                  "Only arithmetic types are supported");

    if constexpr (std::is_floating_point_v<T>) {
      fillRandom(std::uniform_real_distribution<T>(-1.0, 1.0));
    } else if constexpr (std::is_same_v<T, bool>) {
      fillRandom(std::bernoulli_distribution());
    } else if constexpr (std::is_integral_v<T>) {
      constexpr int lowerLimit = std::is_signed_v<T> ? -10 : 0;
      fillRandom(std::uniform_int_distribution<T>(lowerLimit, 10));
    }
    return success();
  }

  return failure();
}

static Error fillWithElements(InputGenerationKind inputGenKind, void *ptr,
                              size_t numElements, TensorDType elementType) {
  LogicalResult fillResult = failure();
  if (elementType == TensorDType::f32)
    fillResult = doFill(inputGenKind, (float *)ptr, numElements);
  if (elementType == TensorDType::kBool)
    fillResult = doFill(inputGenKind, (bool *)ptr, numElements);
  if (elementType == TensorDType::si8)
    fillResult = doFill(inputGenKind, (int8_t *)ptr, numElements);
  if (elementType == TensorDType::ui8)
    fillResult = doFill(inputGenKind, (uint8_t *)ptr, numElements);
  if (elementType == TensorDType::si16)
    fillResult = doFill(inputGenKind, (int16_t *)ptr, numElements);
  if (elementType == TensorDType::ui16)
    fillResult = doFill(inputGenKind, (uint16_t *)ptr, numElements);
  if (elementType == TensorDType::si32)
    fillResult = doFill(inputGenKind, (int32_t *)ptr, numElements);
  if (elementType == TensorDType::ui32)
    fillResult = doFill(inputGenKind, (uint32_t *)ptr, numElements);
  if (elementType == TensorDType::si64)
    fillResult = doFill(inputGenKind, (int64_t *)ptr, numElements);
  if (elementType == TensorDType::ui64)
    fillResult = doFill(inputGenKind, (uint64_t *)ptr, numElements);

  if (failed(fillResult)) {
    std::string err;
    llvm::raw_string_ostream os(err);
    if (inputGenKind == InputGenerationKind::kRandom)
      os << "Random";
    else if (inputGenKind == InputGenerationKind::kOnes)
      os << "Constant ones";
    else if (inputGenKind == InputGenerationKind::kNegOnes)
      os << "Constant negative ones";
    else if (inputGenKind == InputGenerationKind::kZeros)
      os << "Constant zeros";
    else
      llvm_unreachable("Unknown InputGenerationKind");

    os << " input initialization not implemented for this type!\n";
    return {Twine(os.str())};
  }

  return {""};
}

ErrorOr<SmallVector<void *, 3>>
M::initializeInputMemory(InputGenerationKind inputGenKind,
                         const TFLRuntimeModel &model) {
  SmallVector<void *, 3> inputTensorData;
  for (const auto &[shape, eltType] :
       llvm::zip(model.getInputShapes(), model.getInputElementTypes())) {

    // Do the malloc.
    ssize_t size = eltType.getSizeInBytes(shape.getNumElements());
    assert(size != -1);
    inputTensorData.push_back(malloc(size));

    // Do the fill
    size_t numElements = shape.getNumElements();
    M::Error err = fillWithElements(inputGenKind, inputTensorData.back(),
                                    numElements, eltType);

    if (failed(err)) {
      for (auto t : inputTensorData)
        free(t);
      return err;
    }
  }
  return inputTensorData;
}
