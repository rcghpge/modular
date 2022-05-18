//===- TFLInputGeneration.h -----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TFLITE_TOOLS_TFLINPUTGENERATION_H
#define TFLITE_TOOLS_TFLINPUTGENERATION_H

#include "Support/LLVMForwardDecls.h"

namespace M {
class Error;
class TFLiteType;

enum class InputGenerationKind {
  kNone,   // inputs are not generated
  kRandom, // generate inputs with random elements, distribution depends on type
  kZeros,  // generate inputs filled with zeros
  kOnes,   // generate inputs filled with ones
  kNegOnes, // generate inputs filled with negative ones
};

/// This function fills `ptr` with the number of elements of the correct element
/// type.
Error fillWithElements(InputGenerationKind inputGenKind, void *ptr,
                       size_t numElements, const TFLiteType &elementType);

} // namespace M

#endif // TFLITE_TOOLS_TFLINPUTGENERATION_H
