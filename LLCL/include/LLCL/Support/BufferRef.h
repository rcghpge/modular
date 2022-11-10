//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_SUPPORT_BUFFERREF_H
#define LLCL_SUPPORT_BUFFERREF_H

#include "LLCL/Runtime/Allocator.h"
#include "Support/ErrorOr.h"

#include <utility>

namespace M::LLCL {

/// This class models a reference to a raw data buffer. The reference can be
/// "owned" or "borrowed". If "owned" the underlying buffer is destroyed when
/// the reference is destructed. The destrution doesn't happen for "borrowed"
/// reference, and the assmption is that the buffer outlives the reference.
class BufferRef {
public:
  /// This constructs uninitialized BufferRef.
  BufferRef()
      : allocator(nullptr), data(nullptr), size(0), alignment(None),
        storage(kUninit){};

  /// Deallocate underlying buffer only when it is owned.
  ~BufferRef() {
    if (storage == kOwned)
      allocator->deallocateBytes(data, size);
  }

  // Move is allowed for BufferRef.
  BufferRef(BufferRef &&rhs) { *this = std::move(rhs); }
  BufferRef &operator=(BufferRef &&rhs) {
    allocator = std::exchange(rhs.allocator, nullptr);
    data = std::exchange(rhs.data, nullptr);
    size = std::exchange(rhs.size, 0);
    alignment = std::exchange(rhs.alignment, None);
    storage = std::exchange(rhs.storage, kUninit);
    return *this;
  }

  //===--------------------------------------------------------------------===//
  // Creation methods.
  //===--------------------------------------------------------------------===//

  /// Creates a BufferRef by allocating the new object. The allocated object is
  /// deallocated when the BufferRef is destructed.
  static BufferRef createOwned(Allocator *allocator, size_t size,
                               size_t alignment = kPreferredMemoryAlignment) {
    return BufferRef(allocator, size, alignment);
  }

  /// Returns a BufferRef that wraps the given pointer as its data. Destructing
  /// the BufferRef does not deallocate the data.
  static BufferRef createBorrowed(void *data, size_t size,
                                  Optional<size_t> alignment = None) {
    return BufferRef(data, size, alignment);
  }

  //===--------------------------------------------------------------------===//
  // Core APIs for working with a BufferRef.
  //===--------------------------------------------------------------------===//

  /// Returns a shallow copy of BufferRef. The storage kind is set to kBorrowed
  /// regardless of the original BufferRef storage kind.
  BufferRef borrow() { return createBorrowed(data, size, alignment); }

  /// Release the ownership or the underlying buffer to the new BufferRef and
  /// return it.
  BufferRef release() {
    BufferRef ref;
    ref = std::move(*this);
    return ref;
  }

  /// Returns a deep copy of this buffer with ownining reference.
  /// TODO: Instead of taking copyAllocator as an argument, we may want to use
  /// the allocator from the source object, though it will require capturing
  /// of allocator even for the borrow reference.
  BufferRef deepCopy(Allocator *copyAllocator) const {
    BufferRef ref =
        createOwned(copyAllocator, size,
                    alignment ? *alignment : kPreferredMemoryAlignment);
    std::memcpy(ref.data, data, size);
    return ref;
  }

  /// Returns the pointer to the underlying buffer object. Will returns nullptr
  /// if the buffer ref is uninitialized or the buffer is moved out.
  void *getData() { return data; }
  const void *getData() const { return data; }

  /// Returns the size of the underlying buffer object. Will returns 0 if the
  /// buffer ref is uninitialized or the buffer is moved out.
  size_t getSize() const { return size; }

  /// Returns the alignment of the underlying buffer. Will return None if the
  /// alignment is not provided for the borrowed reference.
  Optional<size_t> getAlignment() const { return alignment; }

  /// Describes whether the underlying buffer storage should be deleted
  /// when this object is destructed.
  enum StorageKind { kUninit, kOwned, kBorrowed };

  StorageKind getStorageKind() const { return storage; }

private:
  /// This constructs "borrowed" BufferRef with the pre-allocated object. The
  /// object is not destroyed when the BufferRef is destroyed.
  BufferRef(void *data, size_t size, Optional<size_t> alignment)
      : allocator(nullptr), data(data), size(size), alignment(alignment),
        storage(kBorrowed) {}

  /// This constructs "owned" BufferRef. The underlying buffer is allocated
  /// when the BufferRef is constructed, and destroyed with the BufferRef
  /// destructed.
  BufferRef(Allocator *allocator, size_t size, size_t alignment)
      : allocator(allocator), size(size), storage(kOwned) {
    data = allocator->allocateBytes(size, alignment);
  }

  // Implicit copying is not allowed.
  BufferRef(const BufferRef &rhs) = delete;
  BufferRef &operator=(const BufferRef &rhs) = delete;

  /// If kOwned, the allocator to which the data must be returned. Otherwise
  /// null.
  Allocator *allocator;
  /// Underlying buffer storage.
  void *data;
  /// Size of underlying buffer.
  size_t size;
  /// Alignment of underlying buffer. None if unknown.
  Optional<size_t> alignment;
  /// Whether the underlying buffer should be deleted when this object is
  /// destructed (kOwned), or never deleted (kBorrowed).
  StorageKind storage;
};

} // namespace M::LLCL

#endif // LLCL_SUPPORT_RCBUFFER_H
