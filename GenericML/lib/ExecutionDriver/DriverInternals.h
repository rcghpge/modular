//===- DriverInternals.h --------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines helper functions used within the ExecutionDriver library.
//
//===----------------------------------------------------------------------===//

#ifndef DRIVERINTERNALS_H
#define DRIVERINTERNALS_H

#include "GenericML/ExecutionDriverCLOptions.h"

namespace M {

/// Initialize raw pointers for each model input with the correct data fill.
/// This allocates the pointer, fills in the data, and returns ownership to the
/// caller.
ErrorOr<SmallVector<Tensor, 3>>
initializeInputMemory(InputGenerationKind inputGenKind, const Model *model,
                      Runtime &runtime);

void benchmarkModel(const Model *model, Runtime &runtime,
                    StringRef benchmarkName, InputGenerationKind inputGenKind,
                    BenchmarkScenario scenario, StringRef outputDir,
                    bool benchmarkNoLogPrefix);
} // end namespace M

#endif // DRIVERINTERNALS_H
