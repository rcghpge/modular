//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_COMPILEDRIVER_H
#define GENERICML_DRIVER_COMPILEDRIVER_H

#include "GenericML/GraphCompiler/Passes.h"
#include "LLCL/ForwardDecls.h"
#include "Support/Buffer.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M {

std::optional<std::filesystem::path> getMaxCacheDir();

/// Gets the location for storing externalized weights.
std::optional<std::filesystem::path> getWeightsCacheDir();

/// Checks if caching is enabled to IR.
bool isModelIRCacheEnabled();

} // namespace M

#endif // GENERICML_DRIVER_COMPILEDRIVER_H
