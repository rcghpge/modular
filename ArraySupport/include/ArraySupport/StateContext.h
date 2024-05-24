//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ARRAY_SUPPORT_STATECONTEXT_H
#define ARRAY_SUPPORT_STATECONTEXT_H

#include "ArraySupport/StateSlot.h"

namespace M {

/// A state context holds a number of 'state slots'. The number of state slots
/// is set at construction time and cannot be changed. Each state slot can hold
/// an arbitrary, reference counted payload (which should derive from
/// ReferenceCounted). Generally the state context is created during a model's
/// 'init' phase, each slot is setup using unique slot indexes determined at
/// compile time, and the state context is reused by each call to the model's
/// 'execute' phase.
///
/// State slots may hold state context for specific operations in the model, or
/// may hold framework or runtime specific state shared across many operations
/// in the model. State slots also support generically caching buffer
/// allocations.
///
/// No thread synchronization is enforced. A slot index should be set or
/// emplaced at most once, which is easy to enforce if the slots are setup
/// only in the 'init' phase. A state slot's contents should never be
/// concurrently mutated and/or read, which is easy to enforce if the slot
/// is used by exactly one operation.
///
/// Note that the same ExecutableModel can only safely support concurrent
/// calls to the 'execute' block if all state slots are read only.
/// TODO: GRA-549
class StateContext {
public:
  /// Constructs a new state context with numSlots uninitialized state slots.
  explicit StateContext(size_t numSlots)
      : numSlots(numSlots),
        stateSlots(std::make_unique<StateSlot[]>(numSlots)) {}

  /// Destroys all state slots in the state context. Slots are destroyed in
  /// reverse order since later slots may hold implicit references to earlier
  /// slots.
  ~StateContext() = default;

  /// Returns the number of state slots in the state context.
  size_t getNumStateSlots() const { return numSlots; }

  /// Returns true if the state slot at slotIndex has been set or emplaced.
  bool hasStateSlot(size_t slotIndex) {
    assert(slotIndex < numSlots &&
           "slotIndex must be within bounds when accessing state context");
    return stateSlots[slotIndex];
  }

  // No implicit copying is allowed.
  StateContext(const StateContext &rhs) = delete;
  StateContext &operator=(const StateContext &rhs) = delete;

  /// Moves an already constructed reference to T into the state slot
  /// at slotIndex. The state slot must not have been already set or emplaced.
  /// The state slot will also hold space for numBuffers cached buffer
  /// allocations, which will all initially be null.
  template <typename T>
  T &setStateSlot(size_t slotIndex, RCRef<T> payload, size_t numBuffers = 0) {
    assert(slotIndex < numSlots &&
           "slotIndex must be within bounds when accessing state context");
    assert(!stateSlots[slotIndex] &&
           "slotIndex has already been set or emplaced");
    stateSlots[slotIndex].reset<T>(std::move(payload), numBuffers);
    return stateSlots[slotIndex].get<T>();
  }

  /// Constructs a T in-place within the state slot at slotIndex. The state
  /// slot must not have already been set or emplaced. The state slot will also
  /// hold space for numBuffers cached buffer allocations, which will all
  /// initially be null.
  template <typename T, typename... Args>
  T &emplaceStateSlot(size_t slotIndex, size_t numBuffers, Args &&...args) {
    assert(slotIndex < numSlots &&
           "slotIndex must be within bounds when accessing state context");
    assert(!stateSlots[slotIndex] &&
           "slotIndex has already been set or emplaced");
    stateSlots[slotIndex].reset<T>(
        RCRef<T>::create(std::forward<Args>(args)...), numBuffers);
    return stateSlots[slotIndex].get<T>();
  }

  /// Returns the state slot at slotIndex. The state slot must have already
  /// been emplaced.
  StateSlot &getStateSlot(size_t slotIndex) {
    assert(slotIndex < numSlots &&
           "slotIndex must be within bounds when accessing state context");
    assert(stateSlots[slotIndex] && "slotIndex has not been set or emplaced");
    return stateSlots[slotIndex];
  }

  /// Returns the payload in state slot at slotIndex. The state slot must
  /// have already been emplaced and be of type T.
  template <typename T>
  T &getStateSlotPayload(size_t slotIndex) {
    return getStateSlot(slotIndex).get<T>();
  }

private:
  size_t numSlots;
  std::unique_ptr<StateSlot[]> stateSlots;
};

} // namespace M

#endif // ARRAY_SUPPORT_STATECONTEXT_H
