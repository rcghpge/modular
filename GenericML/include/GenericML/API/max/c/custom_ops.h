//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MODULAR_C_CUSTOM_OPS_H
#define MODULAR_C_CUSTOM_OPS_H

#include "max/c/symbol_export.h"
#include "max/c/types.h"

/// Replaces Modular Kernels with user defined kernels.
///
/// You must call this before you call `M_compileModel()`.
///
/// @param compileConfig The compilation configuration for your model, from
/// `M_newCompileConfig()`.
/// @param path The path to your custom op kernel.
/// @param status The status used to report errors in the case of failures.
MODULAR_API_EXPORT void M_useKernelsFrom(M_CompileConfig *compileConfig,
                                         const char *path, M_Status *status);

#endif // MODULAR_C_EXPERIMENTAL_CUSTOM_OPS_H
