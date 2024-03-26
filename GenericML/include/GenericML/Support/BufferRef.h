//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_BUFFERREF_H
#define GENERICML_SUPPORT_BUFFERREF_H

#include "GenericML/Support/Assert.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/AnyAsyncValueRef.h"
#include "LLCL/Runtime/CompactRuntimePtr.h"
#include "LLCL/Support/Resource.h"
#include "Support/ErrorOr.h"
#include "Support/Profiling/TimeProfiler.h"

#include "llvm/Support/Alignment.h"

#include <utility>

namespace M::GML {

// A block of owned memory. When there are no more references to this it will
// free the underlying memory.
struct OwnedMemoryBlock {
  OwnedMemoryBlock() = default;
  OwnedMemoryBlock(size_t size, std::optional<size_t> alignment,
                   LLCL::CompactRuntimePtr rt);
  ~OwnedMemoryBlock();
  void *data;
  size_t size;

  LLCL::CompactRuntimePtr runtime;
};

/// Profiler entry for tracking the lifetimes of owned buffers.
using BufferRefLifetimeProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kMem, 3), Trace::kMem>;

/// A reference (aka smart pointer) to an underlying buffer of known size but
/// uninterpreted and possibly uninitialized bytes, along with metadata needed
/// to interop with the Modular runtime and free the buffer when required.
///
/// The underlying buffer may be 'owned', 'borrowed' or 'foreign'.
/// An owning reference frees its buffer when destroyed. A 'borrowed' reference
/// does not attempt to free its buffer. A 'foreign' reference holds a
/// 'foreign object' instead of the buffer itself, and a matching
/// ForeignBufferInterface registered with the ForeignBufferInterfaceRegistry
/// supplied methods to extract and (if required) free the underlying buffer.
///
/// References may be moved (the underlying buffer is transferred), explicitly
/// copied (the underlying buffer will be copied), and borrowed (the new
/// reference will contain a buffer alias).
///
/// Zero size buffers are representable and are distinct from the 'null' buffer
/// (built by the default constructor). A zero size buffer still has a non-null
/// buffer pointer, though obviously there's nothing that can be done with that
/// pointer. (We are not sensitive to the system's malloc handling of zero
/// sizes, which could return null or a pointer.)
///
/// Buffers may be marked immutable, however this will only trigger an assert
/// on a call to getMutableBuffer and no deeper protection is enforced.
///
/// All references remember their (optional) alignment constraint, which may be
/// used when the buffer is copied.
///
/// All borrowing references capture a coarse-grained notion of lifetime.
/// This is used to ensure borrows are done safely, particularly when crossing
/// the boundary between Modular and third-party code. If it is difficult to
/// statically guarantee a borrowed buffer will outlive the borrow then an
/// AnyAsyncValueRef may be captured in the buffer reference to keep the
/// object owning the borrowed buffer alive.
///
/// CAUTION: Though movable, this type is 32 bytes on 64-bit systems (without
/// profiling) so best not to consider it a simple handle.
/// WARNING: If the BufferRef holds a 'owningRef' then a move will invoke
/// atomic reference counting.
class BufferRef {
public:
  /// DEPRECATED.
  enum Lifetime : uint8_t {
    kLocal = 0,
    kExecute = 1,
    kConstant = 2,
  };

  /// Constructs a reference in the 'null' state. Useful for references
  /// which are allocated lazily.
  BufferRef() = default;

  /// Deallocates the underlying buffer if an owning reference.
  ~BufferRef();

  // Move is allowed for buffer references. The moved-from reference is
  // returned to the default 'null' state.
  BufferRef(BufferRef &&rhs);
  BufferRef &operator=(BufferRef &&rhs);

  //===--------------------------------------------------------------------===//
  // Creation methods.
  //===--------------------------------------------------------------------===//

  /// Creates a BufferRef with a given storage handle. The handle may be shared
  /// between other buffer refs will last as long as any use is alive. It should
  /// deallocate `buffer` on its destruction.
  static BufferRef create(void *buffer, size_t size,
                          AnyAsyncValueRef storageHandle,
                          std::optional<size_t> alignment);
  static BufferRef create(void *buffer, size_t size,
                          AnyAsyncValueRef storageHandle,
                          llvm::MaybeAlign alignment);

  /// Create a buffer ref which does not track its memory and will not free the
  /// memory.
  static BufferRef createWithNonTrackedMemory(void *buffer, size_t size,
                                              std::optional<size_t> alignment);

  /// Creates an 'owning' reference by allocating a new underlying buffer of
  /// size and alignment. The allocated buffer will be freed when the reference
  /// is destroyed. Since ownership rests with the caller, the lifetime will
  /// be set to 'local' by default. If copyFrom is non-null its contents will be
  /// copied into the new underlying buffer. Otherwise the new underlying buffer
  /// is left uninitialized. It is valid for size to be zero, and the resulting
  /// reference will not be 'null'.
  static BufferRef createOwned(LLCL::CompactRuntimePtr runtime, size_t size,
                               std::optional<size_t> alignment,
                               const void *copyFrom = nullptr);

  //===--------------------------------------------------------------------===//
  // Core APIs for working with a BufferRef.
  //===--------------------------------------------------------------------===//

  /// Returns a borrowing reference to the buffer of this reference.
  /// This reference may be owning or borrowing, and the underlying buffer must
  /// outlive the returned reference. The result will have the given
  /// borrowLifetime.
  ///
  /// Assert fails if asking to borrow from a borrow whose lifetime is
  /// not greater or equal to borrowLifetime.
  BufferRef borrow() const;

  /// Returns an owning reference to a deep copy of this reference.
  /// The given runtime will be used for the copy. Any alignment
  /// constraint in this reference will be respected by the copy.
  BufferRef deepCopy(LLCL::CompactRuntimePtr newRuntime) const;

  /// Returns a pointer to the underlying buffer. Returns nullptr if the
  /// reference is 'null' or has been moved out of. A zero sized buffer
  /// will not return nullptr.
  void *getBuffer() { return data; }
  const void *getBuffer() const { return data; }

  template <typename T>
  T *getBuffer() {
    return static_cast<T *>(getBuffer());
  }
  template <typename T>
  const T *getBuffer() const {
    return static_cast<const T *>(getBuffer());
  }

  /// Returns the size of the underlying buffer. Will return 0 if the
  /// reference is uninitialized or has been moved out of.
  size_t getSize() const { return size; }

  /// Returns the alignment of the underlying buffer, or none if no alignment
  /// constraint was given.
  std::optional<size_t> getAlignment() const {
    return alignment.has_value()
               ? std::optional<size_t>(static_cast<size_t>(alignment->value()))
               : std::nullopt;
  }

  /// If the buffer is owning return the memory block which owns the memory.
  AnyAsyncValueRef getMemStorageHandle() {
    if (!owningRef)
      return {};
    return owningRef.copy();
  }

  void printDebug(raw_ostream &os) const;
  void swap(BufferRef &rhs);

#if MODULAR_PARANOID
  /// Adds to uses a fresh use capturing that this buffer reference is being
  /// read, written or mutated. If this buffer reference is itself a borrow
  /// of some other buffer reference, then uses will be added for the
  /// transitive read/write/mutate, and so on to closure. All these uses should
  /// be held for the duration of the kernel which is performing the access,
  /// but no longer.
  void collectAccessUses(
      SmallVectorImpl<LLCL::ResourceUse> &uses, StringRef name,
      LLCL::ResourceUseType useType,
      const LLCL::ResourceSection &section = LLCL::allResourceSection()) const;
#endif

private:
  /// Constructs a kOwned or kBorrowed reference.
  BufferRef(void *buffer, size_t size, AnyAsyncValueRef owningRef,
            llvm::MaybeAlign alignment,
            BufferRefLifetimeProfilerEntry profilerEntry
#if MODULAR_PARANOID
            ,
            LLCL::ResourceRef resource, LLCL::ResourceUse use
#endif
            )
      : data(buffer), size(size), owningRef(std::move(owningRef)),
        alignment(alignment), profilerEntry(std::move(profilerEntry))
#if MODULAR_PARANOID
        ,
        resource(std::move(resource)), use(std::move(use))
#endif
  {
  }

  // Implicit copying is not allowed.
  BufferRef(const BufferRef &rhs) = delete;
  BufferRef &operator=(const BufferRef &rhs) = delete;

  // Reset the reference to the kNull state. However only the ownership,
  // buffer/object, and size fields are reset.
  void reset();

  /// If kOwned or kBorrowed, the underlying buffer.
  /// If kForeign, a pointer to the 'foreign' object backing the buffer.
  /// If kNull, null.
  void *data = nullptr;

  /// The size of the underlying buffer.
  size_t size = 0;
  /// If kOwned or kNull, null.
  /// If kBorrowed or kForeign, an optional reference to an AsyncValue which
  /// owns or otherwise keeps the borrowed buffer or foreign object alive.
  AnyAsyncValueRef owningRef;

  /// Alignment of underlying buffer, or None if no specific alignment given.
  llvm::MaybeAlign alignment = std::nullopt; // 2 bytes

  BufferRefLifetimeProfilerEntry profilerEntry;

#if MODULAR_PARANOID
  /// An abstract 'resource' to represent this buffer. May be 'used' by
  /// borrows from this buffer.
  LLCL::ResourceRef resource;

  /// If the buffer is kBorrowed, an abstract 'use' of borrowee. Otherwise null.
  LLCL::ResourceUse use;
#endif
};

inline raw_ostream &operator<<(raw_ostream &os, const BufferRef &value) {
  value.printDebug(os);
  return os;
}

} // namespace M::GML

namespace std {

// For ADL style swap.
template <>
inline void swap(M::GML::BufferRef &lhs, M::GML::BufferRef &rhs) {
  lhs.swap(rhs);
}

} // namespace std

#endif // GENERICML_SUPPORT_BUFFERREF_H
