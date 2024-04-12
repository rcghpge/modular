//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ARRAY_SUPPORT_BUFFERREF_H
#define ARRAY_SUPPORT_BUFFERREF_H

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
                   LLCL::CompactRuntimePtr rt, void *takePtr = nullptr);
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
/// to interop with the Modular runtime. Each buffer has an async value to its
/// storage which will be freed on deconstruction.
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
  /// size and alignment.
  static BufferRef createOwned(LLCL::CompactRuntimePtr runtime, size_t size,
                               std::optional<size_t> alignment);
  /// Creates an 'owning' reference of size and alignment using takeFrom void*
  /// ptr that is allocated by caller. There is an assumption that the takeFrom
  /// pointer is allocated by caller via untime->getAllocator(). The allocated
  /// buffer will be freed by BufferRef when the reference is destroyed. Since
  /// the ownership lies with BufferRef, caller should not free anything that
  /// has been taken by BufferRef.
  static BufferRef take(LLCL::CompactRuntimePtr runtime, size_t size,
                        size_t alignment, void *takeFrom);
  //===--------------------------------------------------------------------===//
  // Core APIs for working with a BufferRef.
  //===--------------------------------------------------------------------===//

  /// Returns a new BufferRef pointing to the same data as this buffer. The new
  /// BufferRef extends the storage lifetime of the memory.
  BufferRef borrow() const;

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
    if (!storageRef)
      return {};
    return storageRef.copy();
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
  BufferRef(void *buffer, size_t size, AnyAsyncValueRef storageRef,
            llvm::MaybeAlign alignment,
            BufferRefLifetimeProfilerEntry profilerEntry
#if MODULAR_PARANOID
            ,
            LLCL::ResourceRef resource, LLCL::ResourceUse use
#endif
            )
      : data(buffer), size(size), storageRef(std::move(storageRef)),
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

  /// The data contained in the buffer
  void *data = nullptr;

  /// The size of the underlying buffer.
  size_t size = 0;

  /// Tracks the storage object which preserves the lifetime of data.
  AnyAsyncValueRef storageRef;

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

#endif // ARRAY_SUPPORT_BUFFERREF_H
