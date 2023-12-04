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
namespace M::GML {
class Tensor;
} // namespace M::GML
namespace M::LLCL {
class CompactRuntimePtr;
} // namespace M::LLCL

namespace M::BenchmarkCLI {

// Read a NumPy NPY file out of the given memory buffer.  The returned tensor
// will own the memory buffer.  The tensor data is not copied.
ErrorOr<GML::Tensor>
readNPYFile(LLCL::CompactRuntimePtr runtime,
            std::unique_ptr<llvm::MemoryBuffer> npyFileBuffer);

} // namespace M::BenchmarkCLI

#endif // GENERICML_BENCHMARK_CLI_NPY_READER_H
