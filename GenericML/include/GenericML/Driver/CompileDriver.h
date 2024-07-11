//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_COMPILEDRIVER_H
#define GENERICML_DRIVER_COMPILEDRIVER_H

#include "AsyncRT/ForwardDecls.h"
#include "Cache/CachedTransform.h"
#include "Cache/Support/Keys.h"
#include "GenericML/GraphCompiler/Passes.h"
#include "Support/Buffer.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/Support/Process.h"

namespace M {

std::optional<std::filesystem::path> getMaxCacheDir();

/// Gets the location for storing externalized weights.
std::optional<std::filesystem::path> getWeightsCacheDir();

/// Checks if compilation progress bar is enabled.
bool isCompileProgressEnabled();

/// Checks if caching is enabled to IR.
bool isModelIRCacheEnabled();

/// Given a transformation function that produces MEF buffer this will run the
/// function and store the MEF in cache, and returns the buffer if cache is
/// enabled. If cache is not enabled  we will skip storing to cache and return
/// the buffer.
ErrorOr<BufferRef>
cacheMEFTransform(StringRef frameworkName,
                  llvm::function_ref<ErrorOr<BufferRef>()> loweringFunction,
                  WriteableBufferRef key);

/// A convenience function to create cache key given a file buffer and a set of
/// options. It is assumed that all options contain a method `print`  that takes
/// a raw_ostream by reference to get a representation of it.
template <typename... Options>
WriteableBufferRef makeCacheKey(BufferRef buffer, Options &...options) {
  // Make key for MEF.
  WriteableBufferRef keyBuffer = WriteableBuffer::get();

  /// Add options to key.
  auto insertOptionToKey = [&](auto &option) { option.print(*keyBuffer); };
  (insertOptionToKey(options), ...);

  /// Add source buffer.
  std::string bufferKey =
      Cache::Keys::TypeKey<BufferRef>::hashKey(std::move(buffer));
  keyBuffer->write(bufferKey.data(), bufferKey.size());
  return keyBuffer;
}
} // namespace M

#endif // GENERICML_DRIVER_COMPILEDRIVER_H
