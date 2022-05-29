//===- TFLInputGeneration.h -----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TFLITE_TOOLS_TFLINPUTGENERATION_H
#define TFLITE_TOOLS_TFLINPUTGENERATION_H

#include "GenericML/Support/Tensor.h"
#include "LLCL/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/SmallVector.h"

namespace M {
template <typename T>
class ErrorOr;
class TFLRuntimeModel;
class Tensor;

enum class InputGenerationKind {
  kRandom, // generate inputs with random elements, distribution depends on type
  kZeros,  // generate inputs filled with zeros
  kOnes,   // generate inputs filled with ones
  kNegOnes, // generate inputs filled with negative ones
};

/// Initialize raw pointers for each model input with the correct data fill.
/// This allocates the pointer, fills in the data, and returns ownership to the
/// caller.
ErrorOr<SmallVector<Tensor, 3>>
initializeInputMemory(InputGenerationKind inputGenKind,
                      const TFLRuntimeModel &model, Runtime &runtime);

} // namespace M

#endif // TFLITE_TOOLS_TFLINPUTGENERATION_H
