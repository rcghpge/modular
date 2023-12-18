//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_ENGINECLI_BENCHMARK_API_BRIDGE_H
#define GENERICML_ENGINECLI_BENCHMARK_API_BRIDGE_H

#include "Support/ForwardDecls.h"
#include "modular/cpp/aliases.h"

namespace M {
class TensorSpec;
} // namespace M

namespace M::BenchmarkCLI {

M_Dtype dtypeToCAPI(DType dty);
DType capiToDType(M_Dtype dty);

TensorSpecPtr tensorSpecToCAPI(const char *name, const TensorSpec &spec);
TensorSpec capiToTensorSpec(const M_TensorSpec *spec);

} // namespace M::BenchmarkCLI

#endif // GENERICML_ENGINECLI_BENCHMARK_API_BRIDGE_H
