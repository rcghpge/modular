//===- CompileDriver.cpp --------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/CompileDriver.h"
#include "GraphRT/MLIRToBEF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace M;

std::unique_ptr<llvm::MemoryBuffer> M::convertToBEF(ModuleOp module,
                                                    bool enableBEFMetadata) {
  // Convert to BEF
  auto result = translateMLIRToBEF(module, enableBEFMetadata);
  if (result.empty())
    return {};

  return llvm::MemoryBuffer::getMemBufferCopy(
      StringRef((const char *)result.data(), result.size()));
}
