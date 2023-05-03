//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_BUFFERREF_H
#define GENERICML_SUPPORT_BUFFERREF_H

#include "GenericML/Support/ForeignBufferInterface.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/CompactRuntimePtr.h"
#include "LLCL/Support/Profiling.h"
#include "Support/ErrorOr.h"

#include "llvm/Support/Alignment.h"

#include <utility>

namespace M::GML {

/// If enabled, the lifetime of every owning BufferRef is tracable.
/// Name: "mem.buffer.owned"
using BufferRefLifetimeProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kMem, 2)>;

/// A reference (aka smart pointer) to an underlying buffer of known size but
/// uninterpreted and possibly uninitialized bytes, along with metadata needed
/// to interop with the Modular runtime and free the buffer when required.
///
/// The underlying buffer may be 'owned' or 'borrowed' by this reference.
/// An owning reference frees its buffer when destroyed. It is possible for
/// the 'owned' underlying buffer to be taken from a 'foreign' memory manager
/// provided a matching ForeignBufferInterface has already been registered with
/// the ForeignBufferInterfaceRegistry. We support this without the overhead
/// of vtable pointers in every BufferRef.
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
/// the boundary between Modular and third-party code.
///
/// CAUTION: Though movable, this type is 24 bytes on 64-bit systems so best not
///          to consider it a simple handle.
class BufferRef {
public:
  /// Describes whether the underlying buffer is owned, or borrowed from
  /// some other owner.
  enum Ownership : uint8_t {
    /// The reference is in the null state.
    kNull = 0,
    /// The buffer was allocated from a Modular Runtime and must be freed
    /// when the reference is destroyed.
    kOwned,
    /// The buffer was borrowed from some other owner.
    kBorrowed,
    /// The buffer was allocated by a 'foreign' memory management system and
    /// must be freed by that system when the reference is destroyed.
    /// A registered ForeignBufferInterface must be used to retrieve the
    /// buffer's data, size, and to perform the free.
    kForeign,
  };

  /// Describes a sound approximation of the lifetime of the owner of
  /// (borrowed) buffer. Lifetimes can help determine when buffer copies can
  /// be safely avoided in favor of borrows, and help debug lifetime overhang
  /// bugs (ie use-after-free). Can be compared with >= to test if lhs lifetime
  /// contains or is equivalent to the rhs lifetime.
  enum Lifetime : uint8_t {
    /// The buffer is owned by a reference with some local scope. Any borrowers
    /// are responsible for ensuring lifetimes don't overhang. Typically this
    /// implies all uses of the borrow are clearly scoped under the owners
    /// construction and destruction. Owning references always have this
    /// lifetime.
    kLocal = 0,

    /// The buffer is owned by:
    ///  - a caller to a model's execute entry point which is held alive
    ///    for the duration of the call (as an input or an output).
    ///  - a context (eg within a state slot of an M::GML::Context) held by
    ///    the executor. Though will be retained ofter the execute entry
    ///    point has returned, a following execute call may reallocate or
    ///    otherwise invalidate the buffer.
    kExecute = 1,

    /// (Technically we could have a lifetime to describe the executor itself,
    /// however there doesn't seem to be any need for it.)

    /// The buffer is a constant (eg an attribute in a MEFFile or some other
    /// mmapped memory). The constant may be shared between multiple executors.
    kConstant = 2,
  };

  /// Describes whether the underlying buffer can be mutated.
  enum Mutability : uint8_t {
    kImmutable = 0,
    kMutable,
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

  /// Creates an owning reference by allocating a new underlying buffer of
  /// size and alignment. The allocated buffer will be freed when the reference
  /// is destroyed. Since ownership rests with the caller, the lifetime will
  /// be set to 'local' by default. If copyFrom is non-null its contents will be
  /// copied into the new underlying buffer. Otherwise the new underlying buffer
  /// is left uninitialized. It is valid for size to be zero, and the resulting
  /// reference will not be 'null'.
  static BufferRef createOwned(LLCL::CompactRuntimePtr runtime, size_t size,
                               std::optional<size_t> alignment,
                               const void *copyFrom = nullptr);

  /// Creates a borrowing reference to the given buffer of the given size and
  /// alignment. The alignment will be respected should the buffer need to be
  /// copied. The returned reference must outlive the given buffer. The given
  /// lifetime should be a sound approximation of the lifetime of the buffer.
  /// It is valid for size to be zero, and the resulting reference will not
  /// be 'null' irrespective of buffer argument.
  static BufferRef createBorrowed(void *buffer, size_t size,
                                  std::optional<size_t> alignment,
                                  Lifetime lifetime);
  static BufferRef createBorrowed(const void *buffer, size_t size,
                                  std::optional<size_t> alignment,
                                  Lifetime lifetime);

  /// Slices this buffer by offset and returns a new buffer ref starting from
  /// that pointer of the given size which is a borrow of this buffer.
  BufferRef createBorrowedSlice(size_t offset, size_t size);

  /// Creates an owning reference to the already allocated 'foreign' buffer
  /// object. The corresponding ForeignBufferInterface must have been already
  /// registered under id. The buffer is considered immutable. It is valid for
  /// size to be zero, and the resulting reference will not be 'null'.
  static BufferRef createForeign(ForeignBufferInterfaceID id, void *object,
                                 size_t size, std::optional<size_t> alignment);

  //===--------------------------------------------------------------------===//
  // Core APIs for working with a BufferRef.
  //===--------------------------------------------------------------------===//

  /// Returns a borrowing reference to the buffer of this reference.
  /// This reference may be owning or borrowing, and the underlying buffer
  /// must outlive the returned reference. The result will have the given
  /// borrowLifetime.
  ///
  /// Returns an error if asking to borrow from a borrow whose lifetime is
  /// not greater or equal to borrowLifetime.
  ErrorOr<BufferRef> borrowOrError(Lifetime borrowLifetime) const;

  /// Returns a borrowing reference to the buffer of this reference.
  /// This reference may be owning or borrowing, and the underlying buffer must
  /// outlive the returned reference. The result will have the given
  /// borrowLifetime.
  ///
  /// Assert fails if asking to borrow from a borrow whose lifetime is
  /// not greater or equal to borrowLifetime.
  BufferRef borrow(Lifetime borrowLifetime) const;

  /// Returns an owning reference to a deep copy of this reference.
  /// The given runtime will be used for the copy. Any alignment
  /// constraint in this reference will be respected by the copy.
  BufferRef deepCopy(LLCL::CompactRuntimePtr newRuntime) const;

  /// Returns a pointer to the underlying buffer. Returns nullptr if the
  /// reference is 'null' or has been moved out of. A zero sized buffer
  /// will not return nullptr.
  void *getMutableBuffer() {
    assert((ownership == kNull || mutability == kMutable) &&
           "buffer is immutable");
    assert(ownership != kForeign && "foreign buffers assumed immutable");
    return bufferOrObject;
  }
  const void *getBuffer() const {
    if (ownership == kForeign) // typically false
      return ForeignBufferInterfaceRegistry::getBuffer(nativeOrForeign.id,
                                                       bufferOrObject);
    return bufferOrObject;
  }

  template <typename T>
  T *getMutableBuffer() {
    return static_cast<T *>(getMutableBuffer());
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

  /// Returns the runtime responsible for this owning references, or null if
  /// the reference is null, borrowed or foreign.
  LLCL::CompactRuntimePtr getRuntime() const {
    if (ownership == kForeign)
      return {};
    return nativeOrForeign.runtime;
  }

  Ownership getOwnership() const { return ownership; }
  Lifetime getLifetime() const { return lifetime; }
  Mutability getMutability() const { return mutability; }
  explicit operator bool() const { return ownership != kNull; }

  /// If the buffer is owning, releases ownership of the underlying owned
  /// buffer and returns it. Otherwise returns null and leaves the reference
  /// unchanged.
  void *releaseMutableBufferIfOwned();
  const void *releaseBufferIfOwned();

  /// If the buffer reference is foreign with a the given expectedID as its
  /// ForeignBufferInterfaceID, then releases ownership of the foreign object
  /// and returns it. Otherwise returns null ond leaves the reference unchanged.
  void *maybeReleaseForeignObject(ForeignBufferInterfaceID expectedID);

  void printDebug(raw_ostream &os) const;
  void swap(BufferRef &rhs);

private:
  BufferRef(void *buffer, size_t size, llvm::MaybeAlign alignment,
            Ownership ownership, Lifetime lifetime, Mutability mutability,
            LLCL::CompactRuntimePtr runtime = {})
      : bufferOrObject(buffer), size(size), alignment(alignment),
        ownership(ownership), lifetime(lifetime), mutability(mutability),
        nativeOrForeign(runtime) {
    if (ownership == kOwned)
      profilerEntry =
          BufferRefLifetimeProfilerEntry::create("mem.buffer.owned", [this]() {
            return Twine("size:").concat(Twine(this->size)).concat("B").str();
          });
  }

  BufferRef(ForeignBufferInterfaceID id, void *object, size_t size,
            llvm::MaybeAlign alignment, Ownership ownership, Lifetime lifetime,
            Mutability mutability)
      : bufferOrObject(object), size(size), alignment(alignment),
        ownership(ownership), lifetime(lifetime), mutability(mutability),
        nativeOrForeign(id) {}

  // Implicit copying is not allowed.
  BufferRef(const BufferRef &rhs) = delete;
  BufferRef &operator=(const BufferRef &rhs) = delete;

  // Reset the reference to the kNull state. However only the ownership,
  // buffer/object, and size fields are reset.
  void reset() {
    ownership = kNull;
    bufferOrObject = nullptr;
    size = 0;
    // Though the underlying buffer has not been freed, stop the profiling
    // entry since we can no longer track its lifetime.
    std::move(profilerEntry).record();
  }

  /// If kOwned or kBorrowed, the underlying buffer.
  /// If kForeign, a pointer to the 'foreign' object backing the buffer.
  /// If kNull, null.
  void *bufferOrObject = nullptr;
  /// The size of the underlying buffer.
  size_t size = 0;

  /// Alignment of underlying buffer, or None if no specific alignment given.
  llvm::MaybeAlign alignment = std::nullopt; // 2 bytes
  /// Whether the underlying buffer should be deleted when this object is
  /// destructed (kOwned), or never deleted (kBorrowed).
  Ownership ownership = kNull; // 1 byte
  /// A sound approximation of the lifetime of the buffer, irrespective of
  /// whether it is owned or borrowed.
  Lifetime lifetime = kLocal; // 1 byte
  /// Whether buffer can be mutated.
  Mutability mutability = kImmutable; // 1 byte

  union NativeOrForeign {
    /// If kOwned, the runtime to which the buffer must be returned.
    /// If kNull or kBorrowed, invalid.
    LLCL::CompactRuntimePtr runtime; // 1 byte
    /// If kForeign, ID for registered foreign buffer interface which can
    /// interpret the bufferOrObject field.
    ForeignBufferInterfaceID id; // 1 byte.

    NativeOrForeign() : runtime() {}
    NativeOrForeign(LLCL::CompactRuntimePtr runtime) : runtime(runtime) {}
    NativeOrForeign(ForeignBufferInterfaceID id) : id(id) {}
  };

  NativeOrForeign nativeOrForeign; // 1 byte

  BufferRefLifetimeProfilerEntry profilerEntry;
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
