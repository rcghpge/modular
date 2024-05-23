//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ARRAY_SUPPORT_STATESLOT_H
#define ARRAY_SUPPORT_STATESLOT_H

#include "ArraySupport/BufferRef.h"
#include "LLCL/ForwardDecls.h"
#include "LLCL/Runtime/AnyAsyncValueRef.h"
#include "Support/ADT/GenericRCRef.h"
#include "Support/ErrorOr.h"
#include "Support/TypeID.h"

namespace M::GML {

/// A type-erased wrapper for all context 'state slots'. Buffer allocations
/// may be cached within a number of 'buffer slots' determined when the state
/// slot is constructed. The state slot may also hold an arbitrary reference
/// counted payload (which should derive from ReferenceCounted).
///
/// TODO: Move the 'buffer slots' out of here into a dedicated payload since
/// does not extend to non-CPU buffer types.
class StateSlot {
public:
  /// Constructs the empty state slot.
  StateSlot() = default;

  /// Establishes state slot to take reference to payload. State slot will have
  /// space for numBuffers, initially all null.
  template <typename T>
  void reset(RCRef<T> &&newPayload, size_t numBuffers) {
    payload = GenericRCRef::fromRCRef<T>(std::move(newPayload));
    bufferCache.clear();
    bufferCache.resize(numBuffers);
  }

  /// Destroy this state slot, dropping the reference to its payload.
  ~StateSlot() = default;

  /// Returns true if have a payload.
  operator bool() const { return payload; }

  /// Returns the payload of type T of this state slot. The static type id for T
  /// must match the type id of the state slot.
  template <typename T>
  T &get() {
    assert(payload && "StateSlot has no payload");
    return *payload.get<T>();
  }

  /// Returns a buffer reference to a buffer of size with alignment.
  /// If a buffer with matching size already exists at bufferIndex in this state
  /// slot then it will be returned. Otherwise, a fresh buffer is allocated
  /// and returned, and any previously cached buffer is freed.
  ///
  /// The result will always be a 'borrow' of the underlying cached buffer.
  /// This means:
  ///  - uses of the returned reference must not outlive the overall context
  ///    holding this state slot.
  ///  - no other call to this method for the same state slot and buffer index
  ///    should be made while the returned reference is still in use.
  BufferRef getOrCreateBuffer(size_t bufferIndex,
                              LLCL::CompactRuntimePtr runtime, size_t size,
                              std::optional<size_t> alignment);

  /// Returns a reference to the already allocated and cached buffer at
  /// bufferIndex in this state slot. Returns Error if no buffer is cached at
  /// given index.
  ///
  /// The result will always be a 'borrow' of the underlying cached buffer.
  /// See getOrCreateBuffer for safety conditions.
  ErrorOr<BufferRef> getBuffer(size_t bufferIndex);

  /// Sets the cached buffer at bufferIndex to the given bufferRef. Returns
  /// error if bufferIndex already contains a buffer.
  ///
  /// If bufferRef is owned we simply take over that ownership. However if
  /// it is borrowed then it must have lifetime of at least 'execute'
  /// so that similarly 'execute'-scoped borrows may be taken from it.
  ErrorOrSuccess setBuffer(size_t bufferIndex, BufferRef bufferRef);

  /// Returns the type id for the value held by this state slot.
  M::TypeID getTypeID() const { return payload.getTypeID(); }

private:
  /// Ensure the buffer in entry is allocated with size, reallocating
  /// any existing buffer if required.
  static void updateCacheIfNeeded(BufferRef &entry,
                                  LLCL::CompactRuntimePtr runtime, size_t size,
                                  std::optional<size_t> alignment);

  /// The state slot payload.
  GenericRCRef payload;

  /// A place to hold pre-allocated or cached buffers. Tensors and other
  /// more refined types can be constructed from these buffers. Though these
  /// references will be 'uninitialized' or 'owning', all access to them
  /// will be via 'borrowed' references. Any 'owning' buffers will be
  /// destroyed when the context holding this state slot is destroyed.
  SmallVector<BufferRef, 1> bufferCache;
};

/// A state slot with no extra payload state. Useful for testing and for
/// ops which have no extra state but wish to make use of buffer caching.
class DefaultStateSlot : public ReferenceCounted<DefaultStateSlot> {};

/// A 'mutable borrow' of a state slot payload of type T. This object holds
/// both the raw pointer to the state slot payload, as well as an async
/// value reference to the outer context for the slot itself. This allows
/// usage of the state slot payload to safely overhang the overall context.
///
/// TODO(#6327): We'd prefer to ensure all computation depending on the
/// outer context be completed before the execute body is complete, using
/// chains where necessary.
template <typename T>
class StateSlotPayloadRef {
public:
  StateSlotPayloadRef(AnyAsyncValueRef outer, T *payload)
      : outer(std::move(outer)), payload(payload) {
    assert(payload && "StateSlotPayloadRef payload cannot be null");
  }

  T &operator*() const { return *payload; }
  T *operator->() const { return payload; }
  T *get() const { return payload; }

  StateSlotPayloadRef copy() const {
    return StateSlotPayloadRef(outer.copy(), payload);
  }

private:
  AnyAsyncValueRef outer;
  T *payload;
};

} // namespace M::GML

#endif // ARRAY_SUPPORT_STATESLOT_H
