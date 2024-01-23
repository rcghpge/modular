//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/CompileDriver.h"
#include "GraphRT/MLIRToMEF.h"
#include "Support/ADT/ConcatenationTree.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace M;

LogicalResult M::convertToMEF(ModuleOp module, raw_ostream &os,
                              bool enableMEFMetadata) {
  ConcatenationTree result = translateMLIRToMEF(module, enableMEFMetadata);
  if (result.getSize() == 0)
    return failure();

  // Write the ConcatenationTree to the output stream.
  result.traverse([&](ArrayRef<uint8_t> data) {
    os.write(reinterpret_cast<const char *>(data.data()), data.size());
  });

  return success();
}
