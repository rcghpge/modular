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
#include "llvm/ADT/StringRef.h"
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
/// exclusions.
MODULAR_VISIBILITY_EXPORT std::optional<PoisonPattern>
getPoisonPattern(const TensorSpec &spec);

/// Fills `totalBytes` bytes of host-visible memory at `ptr` with `pattern`.
MODULAR_VISIBILITY_EXPORT void fillHostMemory(void *ptr, size_t totalBytes,
                                              const PoisonPattern &pattern);

/// What the debug allocator does with one allocation.
struct AllocationPoisonDecision {
  uint64_t ordinal; ///< Position in the current epoch, counted from 0.
  /// The poison pattern for a selected allocation, zero for an unselected
  /// one. Empty leaves the contents alone, which happens under `sync-only`
  /// and for a selected dtype that has no poison pattern.
  std::optional<PoisonPattern> fill;
};

/// Decides how to fill one allocation, consumes an ordinal for it, and logs
/// the decision with that ordinal. `hostAlloc` only labels the log line.
///
/// Allocations that are *not* selected are zero-filled rather than left
/// alone, which is what makes the selector usable as a bisection tool: both
/// arms of a comparison stay deterministic, so a difference in output is
/// attributable to the allocations the selection moved and to nothing else.
/// A *selected* allocation whose dtype has no poison pattern (fp8, sub-byte
/// floats) is left alone, as it was before the selector existed. Zeroing it
/// would make it identical in both arms, so no bisection could implicate it;
/// the price is that a selection containing one is not fully deterministic.
///
/// The selection comes from `MODULAR_DEBUG_ALLOC_POISON_SELECT` (config key
/// `debug.alloc_poison_select`). It is either one of these modes on its own:
///
///   (unset) / `all`    poison every allocation -- the default, and what the
///                      allocator did before the selector existed
///   `none`             poison nothing; zero-fill everything
///   `sync-only`        write nothing at all, but keep the synchronization
///                      after every device allocation
///
/// or one or both of these filters, joined by `&`, all of which must match:
///
///   `dtype=si64,ui32`  poison only these element types, spelled as `DType`
///                      prints them
///   `ord=100-2000`     poison only these allocation ordinals, inclusive;
///                      `ord=100` selects one
///
/// so `dtype=si64&ord=100-2000` poisons the si64 allocations among ordinals
/// 100 to 2000. The separator is not `;` because `modular.cfg` reads a `;` as
/// the start of a comment.
///
/// A malformed clause is ignored whole, with a warning, so a typo widens the
/// selection rather than narrowing it. A mode combined with other clauses, or
/// a second `dtype=` or `ord=`, counts as malformed. A clause that silently
/// selected nothing would produce a clean-looking run and read as "no
/// uninitialized reads here".
///
/// `sync-only` exists because enabling this allocator changes two things at
/// once: it fills memory, and it synchronizes the device after every
/// allocation. On a model that allocates ~190,000 times per run, that second
/// effect serializes host against device almost everywhere, so "the poison
/// allocator makes the output deterministic" is equally explained by the
/// synchronization as by the fill. `sync-only` holds the synchronization and
/// drops the fill, which separates them: still deterministic means a race the
/// synchronization hides, back to non-deterministic means the fill was doing
/// the work and the read is genuinely of uninitialized memory. `none` cannot
/// answer this, because zero-filling makes an uninitialized read deterministic
/// too.
///
/// `MODULAR_DEBUG_ALLOC_POISON_SELECT_FILE` (`debug.alloc_poison_select_file`)
/// names a file holding the same grammar, checked on every allocation. Each
/// change starts a new epoch and **resets ordinals to zero**, so a caller
/// that rewrites the file between two requests can address "the n-th
/// allocation of this request" rather than of this process. That is what
/// lets a bisection run inside one boot: on this workload most allocations
/// recur per execution, so rebooting per step would cost half an hour to
/// change one number. A change is a new modification time, size or file
/// identity rather than new content, because repeating a measurement needs
/// the same ordinals, not merely the same selection. Modification times can
/// be too coarse to tell two quick in-place writes apart, so write the new
/// selection to a temporary file and rename it over this one: a rename
/// always changes the file's identity.
///
/// Ordinals are stable across runs only if the process allocates in the same
/// order each time, which depends on host threading as well as on the
/// device. The `DEBUG_ALLOC_POISON` log line reports every ordinal with its
/// dtype and size, so diffing two runs' logs shows whether they line up, and
/// a log from one run is the index for selecting ranges in the next.
MODULAR_VISIBILITY_EXPORT AllocationPoisonDecision
decidePoison(const TensorSpec &spec, bool hostAlloc);

/// Evaluates `selectionSpec` directly instead of the configured selection,
/// consuming no ordinal and logging nothing, so the selection grammar can be
/// tested without a process per case.
MODULAR_VISIBILITY_EXPORT AllocationPoisonDecision decidePoisonForTesting(
    llvm::StringRef selectionSpec, const TensorSpec &spec, uint64_t ordinal);

/// Points the selection machinery at the selection file `path` and clears its
/// cached state, so the file mechanism can be tested without controlling the
/// environment before static initialization. An empty `path` re-reads the
/// configured selection instead. Not for production use.
MODULAR_VISIBILITY_EXPORT void
resetPoisonSelectionForTesting(llvm::StringRef path);

} // namespace M::GraphCompiler::Driver

#endif // GRAPHCOMPILER_DRIVER_DEVICE_DEBUGALLOCPOISON_H
