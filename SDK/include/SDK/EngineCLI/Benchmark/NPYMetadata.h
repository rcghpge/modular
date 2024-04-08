//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_ENGINECLI_BENCHMARK_NPY_METADATA_H
#define SDK_ENGINECLI_BENCHMARK_NPY_METADATA_H

#include "Support/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"

namespace llvm::json {
class Value;
} // namespace llvm::json

namespace M::BenchmarkCLI {

ErrorOr<llvm::json::Value> parseNPYMetadata(StringRef input);

} // namespace M::BenchmarkCLI

#endif // SDK_ENGINECLI_BENCHMARK_NPY_METADATA_H
