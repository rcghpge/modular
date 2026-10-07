//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_CUDAGDB_H
#define KGEN_TOOLS_MOJO_COMMON_CUDAGDB_H

#include "Support/Driver/DriverSupport.h"
#include "Support/ErrorOr.h"

namespace M {

/// Invokes an cuda-gdb process with the provided arguments.
int invokeCudaGdb(const State &state, ArrayRef<std::string> cudaGdbArgs,
                  ArrayRef<std::string> runArgs = {},
                  std::optional<std::string> cudaGdbPathGiven = std::nullopt,
                  bool breakOnLaunch = false, bool dryRun = false);

} // namespace M

#endif // KGEN_TOOLS_MOJO_COMMON_CUDAGDB_H
