//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_BENCHMARK_CLI_INPUT_GENERATION_DEFAULTS_H
#define GENERICML_BENCHMARK_CLI_INPUT_GENERATION_DEFAULTS_H

#include "modular/c/types.h"

namespace M::BenchmarkCLI {

struct InputGenerationSchema;
struct InputGenerationInput;

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

#endif // GENERICML_BENCHMARK_CLI_INPUT_GENERATION_DEFAULTS_H
