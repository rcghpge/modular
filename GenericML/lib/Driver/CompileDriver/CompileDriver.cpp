//===- CompileDriver.cpp --------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/CompileDriver.h"
#include "GraphRT/MLIRToBEF.h"
#include "Support/ConcatenationTree.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace M;

std::unique_ptr<llvm::MemoryBuffer> M::convertToBEF(ModuleOp module,
                                                    bool enableBEFMetadata) {
  // Convert to BEF
  ConcatenationTree result = translateMLIRToBEF(module, enableBEFMetadata);
  auto resultSize = result.getSize();
  if (resultSize == 0)
    return {};

  // If successful, linearize it into a MemoryBuffer.
  auto buf = llvm::WritableMemoryBuffer::getNewUninitMemBuffer(resultSize);
  // MemoryBuffer likes char, we like uint8_t, adapt.
  auto buffer = MutableArrayRef<uint8_t>((uint8_t *)buf->getBufferStart(),
                                         (uint8_t *)buf->getBufferEnd());

  // Fill in the data by copying it from the ConcatenationTree into the result
  // buffer.
  result.traverse([&](ArrayRef<uint8_t> data) {
    assert(buffer.size() >= data.size() &&
           "ConcatenationTree has unexpected length!");
    // Copy this chunk into the MemoryBuffer.
    std::uninitialized_copy(data.begin(), data.end(), buffer.begin());

    // Consume this space.
    buffer = buffer.drop_front(data.size());
  });

  assert(buffer.empty() && "Should have filled the whole buffer");
  return std::move(buf);
}
