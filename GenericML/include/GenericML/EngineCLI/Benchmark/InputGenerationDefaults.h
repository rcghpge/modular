//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_ENGINECLI_BENCHMARK_INPUT_GENERATION_DEFAULTS_H
#define GENERICML_ENGINECLI_BENCHMARK_INPUT_GENERATION_DEFAULTS_H

#include "modular/c/types.h"

namespace M::EngineCLI {
struct InputGenerationSchema;
struct InputGenerationInput;
} // namespace M::EngineCLI

namespace M::BenchmarkCLI {

// TODO: Reparent BenchmarkCLI so this using is no longer necessary.
using namespace M::EngineCLI;

void applyInputGenerationDefaults(InputGenerationSchema &schema,
                                  const M_AsyncCompiledModel *model,
                                  M_Status *status);
void applyInputListDefaults(InputGenerationSchema &schema,
                            const M_AsyncCompiledModel *model,
                            M_Status *status);
void applyInputShapeDefaults(InputGenerationInput &inputSpec,
                             const M_AsyncCompiledModel *model,
                             M_Status *status);
void applyInputDataDefaults(InputGenerationInput &inputSpec);

} // namespace M::BenchmarkCLI

#endif // GENERICML_ENGINECLI_BENCHMARK_INPUT_GENERATION_DEFAULTS_H
