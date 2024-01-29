//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/CompileDriver.h"
#include "GraphRT/MLIRToMEF.h"
#include "Support/ADT/ConcatenationTree.h"
#include "Support/ML/SizeUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace M;

ErrorOr<M::BufferRef> M::convertToMEF(ModuleOp module, bool enableMEFMetadata) {
  ConcatenationTree result = translateMLIRToMEF(module, enableMEFMetadata);
  if (result.getSize() == 0)
    return Error("failed to convert model to MEF.");

  // Write the ConcatenationTree to the output stream.
  M::WriteableBufferRef outputBuffer = WriteableBuffer::get(
      0, maxAttributeAlignment().value(), result.getSize());
  result.traverse([&](ArrayRef<uint8_t> data) {
    outputBuffer->write(reinterpret_cast<const char *>(data.data()),
                        data.size());
  });

  return outputBuffer;
}
