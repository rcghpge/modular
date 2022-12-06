//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_BUFFERREF_H
#define GENERICML_SUPPORT_BUFFERREF_H

#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/CompactRuntimePtr.h"
#include "Support/ErrorOr.h"

#include "llvm/Support/Alignment.h"

#include <utility>

namespace M::GML {

/// A smart pointer, aka reference, to an underlying buffer of known size but
/// uninterpreted and possibly uninitialized bytes.
///
/// The underlying buffer may be 'owned' or 'borrowed' by this reference.
/// An owning reference frees its buffer when destroyed. References may be
/// moved (the underlying buffer is transferred), explicitly copied (the
/// underlying buffer will be copied), and borrowed (the new reference
/// will contain a buffer alias).
///
/// All owning references remember their runtime, which is needed when owning
/// references are destroyed, and may be used when the buffer is copied. All
/// references remember their (optional) alignment constraint, which may be
/// used when the buffer is copied.
class BufferRef {
public:
  /// Constructs an uninitialized buffer reference. Useful for references
  /// which are allocated lazily.
  BufferRef() = default;

  /// Deallocates underlying buffer if an owning reference.
  ~BufferRef();

  // Move is allowed for buffer references.
  BufferRef(BufferRef &&rhs);
  BufferRef &operator=(BufferRef &&rhs);

  //===--------------------------------------------------------------------===//
  // Creation methods.
  //===--------------------------------------------------------------------===//

  /// Creates an owning reference by allocating a new and uninitialized
  /// underlying buffer of size and alignment. The allocated buffer is freed
  /// when the reference is destroyed.
  /// TOOO(#5799): We're requiring all callers to be explicit about alignment
  /// (or lack thereof) to help track down 'forgotten alignment' problems.
  static BufferRef createOwned(LLCL::CompactRuntimePtr runtime, size_t size,
                               Optional<size_t> alignment) {
    return BufferRef(runtime, size, alignment);
  }

  /// Creates a borrowing reference to the given buffer of the given size and
  /// alignment. The returned reference must outlive the given buffer.
  /// The alignment will be respected should the buffer need to be copied.
  /// TODO(#5369): Also support const void* overload
  static BufferRef createBorrowed(void *buffer, size_t size,
                                  Optional<size_t> alignment) {
    return BufferRef(buffer, size, alignment);
  }

  //===--------------------------------------------------------------------===//
  // Core APIs for working with a BufferRef.
  //===--------------------------------------------------------------------===//

  /// Returns a borrowing reference to the buffer of this reference. This
  /// reference may be owning or borrowing, and the underlying buffer must
  /// outlive the returned reference.
  /// TODO(#5369): Consistent naming with Tensor borrowing.
  BufferRef borrow() const;

  /// Returns an owning reference to a deep copy of this reference. The given
  /// runtime will be used for the copy. Any alignment constraint in this
  /// reference will be respected by the copy.
  BufferRef deepCopy(LLCL::CompactRuntimePtr newRuntime) const;

  /// Returns a pointer to the underlying buffer. Returns nullptr if the
  /// reference is uninitialized or has been moved out of.
  /// TODO(#5369): Rename to getMutableBuffer/getBuffer.
  void *getData() { return buffer; }
  const void *getData() const { return buffer; }

  /// Returns the size of the underlying buffer. Will return 0 if the
  /// reference is uninitialized or has been moved out of.
  size_t getSize() const { return size; }

  /// Returns the alignment of the underlying buffer, or none if no alignment
  /// constraint was given.
  Optional<size_t> getAlignment() const {
    return alignment.has_value()
               ? Optional<size_t>(static_cast<size_t>(alignment->value()))
               : llvm::None;
  }

  /// Returns the alignment of the underlying buffer, or 1 if no alignment
  /// constraint was given.
  size_t getAlignmentOrOne() const { return alignment.valueOrOne().value(); }

  /// Returns the runtime responsible for this owning references, or null if
  /// the reference is a borrow.
  LLCL::CompactRuntimePtr getRuntime() const { return runtime; }

  /// Describes whether the underlying buffer should be deleted when this
  /// reference is destroyed.
  /// TODO(#5369): Will probably extend kBorrowed with lifetime qualifier
  /// to help protect against overhanging borrows.
  enum StorageKind : uint8_t { kUninit, kOwned, kBorrowed };

  StorageKind getStorageKind() const { return storage; }

private:
  /// Constructs an "owned" BufferRef. The underlying buffer is allocated
  /// when the BufferRef is constructed, and destroyed when the BufferRef
  /// is destroyed.
  BufferRef(LLCL::CompactRuntimePtr runtime, size_t size,
            Optional<size_t> alignment);

  /// Constructs a "borrowed" BufferRef with the pre-allocated buffer. The
  /// buffer is not destroyed when the BufferRef is destroyed.
  BufferRef(void *buffer, size_t size, Optional<size_t> alignment);

  // Implicit copying is not allowed.
  BufferRef(const BufferRef &rhs) = delete;
  BufferRef &operator=(const BufferRef &rhs) = delete;

  /// The underlying buffer.
  void *buffer = nullptr;
  /// The size of the underlying buffer.
  size_t size = 0;
  /// The runtime responsible for the underlying buffer if it is owned.
  LLCL::CompactRuntimePtr runtime = nullptr; /* 1 byte */
  /// Alignment of underlying buffer, or None if no specific alignment given.
  llvm::MaybeAlign alignment = {}; /* 2 bytes */
  /// Whether the underlying buffer should be deleted when this object is
  /// destructed (kOwned), or never deleted (kBorrowed).
  StorageKind storage = kUninit; /* 1 byte */
  // TODO(#5369): A notion of read-only (in DEBUG mode) with asserts.
  /* 4 bytes unused on 64-bit systems. */
};

} // namespace M::GML

#endif // GENERICML_SUPPORT_RCBUFFER_H
