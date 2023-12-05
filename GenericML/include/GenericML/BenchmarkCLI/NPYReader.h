//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_BENCHMARK_CLI_NPY_READER_H
#define GENERICML_BENCHMARK_CLI_NPY_READER_H

#include "Support/ForwardDecls.h"
#include <memory>

namespace llvm {
class MemoryBuffer;
} // namespace llvm

namespace M::BenchmarkCLI {

class Tensor;

// Read a NumPy NPY file out of the given memory buffer.  The returned tensor
// will own the memory buffer.  The tensor data is not copied.
ErrorOr<Tensor> readNPYFile(std::unique_ptr<llvm::MemoryBuffer> npyFileBuffer);

} // namespace M::BenchmarkCLI

#endif // GENERICML_BENCHMARK_CLI_NPY_READER_H
