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
template <typename T>
class ErrorOr;
class TFLiteType;
class TFLRuntimeModel;

enum class InputGenerationKind {
  kNone,   // inputs are not generated
  kRandom, // generate inputs with random elements, distribution depends on type
  kZeros,  // generate inputs filled with zeros
  kOnes,   // generate inputs filled with ones
  kNegOnes, // generate inputs filled with negative ones
};

/// Initialize raw pointers for each model input with the correct data fill.
/// This allocates the pointer, fills in the data, and returns ownership to the
/// caller.
ErrorOr<SmallVector<void *, 3>>
initializeInputMemory(InputGenerationKind inputGenKind,
                      const TFLRuntimeModel &model);

} // namespace M

#endif // TFLITE_TOOLS_TFLINPUTGENERATION_H
