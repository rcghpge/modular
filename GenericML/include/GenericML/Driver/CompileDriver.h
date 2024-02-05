//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_COMPILEDRIVER_H
#define GENERICML_DRIVER_COMPILEDRIVER_H

#include "Cache/CachedTransform.h"
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

/// Given a transformation function that produces MEF buffer this will run the
/// function and store the MEF in cache, and returns the buffer if cache is
/// enabled. If cache is not enabled  we will skip storing to cache and return
/// the buffer.
ErrorOr<BufferRef>
cacheMEFTransform(StringRef frameworkName,
                  llvm::function_ref<ErrorOr<BufferRef>()> loweringFunction,
                  WriteableBufferRef key, LLCL::Runtime &runtime);

} // namespace M

#endif // GENERICML_DRIVER_COMPILEDRIVER_H
