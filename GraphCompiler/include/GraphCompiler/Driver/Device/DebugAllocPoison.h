//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHCOMPILER_DRIVER_DEVICE_DEBUGALLOCPOISON_H
#define GRAPHCOMPILER_DRIVER_DEVICE_DEBUGALLOCPOISON_H

#include "Support/ML/DType.h"
#include "Support/ML/TensorSpec.h"
#include "Support/SymbolExport.h"
#include <cstddef>
#include <cstdint>
#include <optional>

namespace M::GraphCompiler::Driver {

/// The dtype-aware bit pattern the debug allocator
/// (MODULAR_DEBUG_DEVICE_ALLOCATOR=uninitialized-poison) uses to fill newly
/// allocated memory, so reads of uninitialized memory can be detected.
struct PoisonPattern {
  uint64_t bits;
  size_t patternBytes;
  const char *patternKind; // "largest-finite" or "byte-fill"; for diagnostics.
};

/// Determines the poison-fill pattern for `spec`'s element type. Returns
/// `std::nullopt` for dtypes excluded from the poison-check set: sub-byte
/// float types (e.g. `float4_e2m1fn`, whose 4-bit lanes can't be written at
/// byte granularity) and fp8 formats (either no `max_finite` value exists, or
/// the narrow dynamic range collides with legitimate saturated quantization
/// values). The Mojo poison-check set is kept in agreement with these
/// exclusions. Logs the skip reason to stderr when returning `std::nullopt`.
MODULAR_VISIBILITY_EXPORT std::optional<PoisonPattern>
getPoisonPattern(const TensorSpec &spec);

/// Fills `spec.getSizeInBytes()` bytes of host-visible memory at `ptr` with
/// the poison pattern from `getPoisonPattern`, for the debug allocator
/// (MODULAR_DEBUG_DEVICE_ALLOCATOR=uninitialized-poison). No-op if `spec`'s
/// dtype has no pattern. `ptr` must point to at least
/// `spec.getSizeInBytes()` writable bytes.
MODULAR_VISIBILITY_EXPORT void
poisonUninitializedMemory(void *ptr, const TensorSpec &spec);

} // namespace M::GraphCompiler::Driver

#endif // GRAPHCOMPILER_DRIVER_DEVICE_DEBUGALLOCPOISON_H
