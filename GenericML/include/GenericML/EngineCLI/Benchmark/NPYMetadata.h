//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_ENGINECLI_BENCHMARK_NPY_METADATA_H
#define GENERICML_ENGINECLI_BENCHMARK_NPY_METADATA_H

#include "Support/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"

namespace llvm::json {
class Value;
} // namespace llvm::json

namespace M::BenchmarkCLI {

ErrorOr<llvm::json::Value> parseNPYMetadata(StringRef input);

} // namespace M::BenchmarkCLI

#endif // GENERICML_ENGINECLI_BENCHMARK_NPY_METADATA_H
