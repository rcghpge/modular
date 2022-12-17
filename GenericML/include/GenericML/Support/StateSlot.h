//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_STATESLOT_H
#define GENERICML_SUPPORT_STATESLOT_H

#include "GenericML/Support/BufferRef.h"
#include "Support/AlignedAlloc.h"
#include "Support/ErrorOr.h"

#include <atomic>
#include <limits>
#include <vector>

namespace M::GML {

namespace Detail {
template <typename T>
class ConcreteStateSlot;
}

/// A type-erased wrapper for all context 'state slots'. Buffer allocations
/// may be cached within a number of 'buffer slots' determined when the state
/// slot is constructed. The state slot may also hold an arbitrary payload,
/// however the payload type must be registered at least once. The payload's
/// destructor will be called when the state slot is destroyed.
class StateSlot {
public:
  /// Every registered payload type is associated with a unique value
  /// of this type.
  using TypeID = uint16_t;

  virtual ~StateSlot() = default;

  // No implicit copying is allowed.
  StateSlot(const StateSlot &rhs) = delete;
  StateSlot &operator=(const StateSlot &rhs) = delete;

  /// Registers type T as being valid for storage within a state slot.
  /// It's ok for the same type to be registered multiple times.
  template <typename T>
  static void registerType() {
    Detail::ConcreteStateSlot<T>::registerType();
  }

  /// Returns a new instance of StateSlot with a freshly constructed payload of
  /// type T which can hold numBuffers cached buffer allocations.
  template <typename T, typename... Args>
  static std::unique_ptr<StateSlot> createStateSlot(size_t numBuffers,
                                                    Args &&...args) {
    return std::make_unique<Detail::ConcreteStateSlot<T>>(
        numBuffers, std::forward<Args>(args)...);
  }

  /// Returns the payload of type T of this state slot. The static type id for T
  /// must match the type id of the state slot.
  template <typename T>
  T &get() {
    assert(typeID == Detail::ConcreteStateSlot<T>::getStaticTypeID() &&
           "mismatched static and dynamic StateSlot types");
    return static_cast<Detail::ConcreteStateSlot<T> *>(this)->payload;
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

protected:
  /// The invalid type id.
  static constexpr TypeID kInvalidTypeID = std::numeric_limits<TypeID>::max();

  StateSlot(TypeID typeID, size_t numBuffers);

  /// Assigns to staticTypeId a globally unique type id.
  static void doTypeRegistration(std::atomic<TypeID> *staticTypeID);

private:
  /// Ensure the buffer in enry is allocated with size, reallocating
  /// any existing buffer if required.
  static void updateCacheIfNeeded(BufferRef &entry,
                                  LLCL::CompactRuntimePtr runtime, size_t size,
                                  std::optional<size_t> alignment);

  /// Type id uniquely describing the type of payload held by this states slot.
  TypeID typeID;

  /// A place to hold pre-allocated or cached buffers. Tensors and other
  /// more refined types can be constructed from these buffers. Though these
  /// references will be 'uninitialized' or 'owning', all access to them
  /// will be via 'borrowed' references. Any 'owning' buffers will be
  /// destroyed when the context holding this state slot is destroyed.
  std::vector<BufferRef> bufferCache;
};

/// A state slot with no extra payload state. Useful for testing and for
/// ops which have no extra state but wish to make use of buffer caching.
class DefaultStateSlot {};

namespace Detail {

/// A state slot holding a payload of type T. This class should never be
/// directly accessed by users, and all access to the underlying T should be
/// via members of StateSlot.
template <typename T>
class ConcreteStateSlot : public StateSlot {
public:
  template <typename... Args>
  explicit ConcreteStateSlot(size_t numOutputs, Args &&...args)
      : StateSlot(getStaticTypeID(), numOutputs),
        payload(std::forward<Args>(args)...) {}

  ~ConcreteStateSlot() override = default;

  // No implicit copying is allowed
  ConcreteStateSlot(const ConcreteStateSlot &rhs) = delete;
  ConcreteStateSlot &operator=(const ConcreteStateSlot &rhs) = delete;

private:
  /// Determines a unique type id to represent T. It's ok for the same
  /// type to be registered multiple times.
  static void registerType() {
    if (getStaticTypeID() != kInvalidTypeID)
      return;
    doTypeRegistration(&staticTypeID);
  }

  /// Returns the unique type representing T. Will be invalid until
  /// registerType is called at least once.
  static TypeID getStaticTypeID() {
    return staticTypeID.load(std::memory_order_acquire);
  }

  /// Unique type id representing T.
  static inline std::atomic<TypeID> staticTypeID{kInvalidTypeID};

  /// Payload.
  T payload;

  // Give StateSlot access to the typeID machinery.
  friend class StateSlot;
};

} // namespace Detail

} // namespace M::GML

#endif // GENERICML_SUPPORT_STATESLOT_H
