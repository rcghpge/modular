//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_CONTEXT_H
#define GENERICML_SUPPORT_CONTEXT_H

#include "GenericML/Support/StateSlot.h"
#include "Support/AlignedAlloc.h"

namespace M::GML {

/// A context holds a number of 'state slots'. The number of state slots is
/// set at construction time and cannot be changed. Each state slot can hold
/// an arbitrary C++ type. Generally the context is created during a model's
/// 'init' phase, each slot is setup using unique slot indexes determined at
/// compile time, and the context is reused by each call to the model's
/// 'execute' phase.
///
/// State slots may hold context for specific operations in the model, or
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
class Context {
public:
  /// Constructs a new context with numSlots uninitialized state slots.
  explicit Context(size_t numSlots) : stateSlots(numSlots) {}
  ~Context() = default;

  /// Returns the number of state slots in the context.
  size_t getNumStateSlots() const { return stateSlots.size(); }

  /// Returns true if the state slot at slotIndex has been set or emplaced.
  bool hasStateSlot(size_t slotIndex) {
    assert(slotIndex < stateSlots.size() && "slotIndex is out of range");
    return stateSlots[slotIndex];
  }

  // No implicit copying is allowed.
  Context(const Context &rhs) = delete;
  Context &operator=(const Context &rhs) = delete;

  /// Moves an already constructed (unique pointer to) T into the state slot
  /// at slotIndex. The state slot must not have been already set or emplaced.
  /// The state slot will also hold space for numBuffers cached buffer
  /// allocations, which will all initially be null.
  template <typename T>
  T &setStateSlot(size_t slotIndex, std::unique_ptr<T> payload,
                  size_t numBuffers = 0) {
    assert(!stateSlots[slotIndex] &&
           "StateSlot for the given slot has already been set or emplaced.");
    stateSlots[slotIndex].reset<T>(std::move(payload), numBuffers);
    return stateSlots[slotIndex].get<T>();
  }

  /// Constructs a T in-place within the state slot at slotIndex. The state
  /// slot must not have already been set or emplaced. The state slot will also
  /// hold space for numBuffers cached buffer allocations, which will all
  /// initially be null.
  template <typename T, typename... Args>
  T &emplaceStateSlot(size_t slotIndex, size_t numBuffers, Args &&...args) {
    assert(!stateSlots[slotIndex] &&
           "StateSlot for the given slot has already been set or emplaced.");
    stateSlots[slotIndex].reset<T>(
        std::make_unique<T>(std::forward<Args>(args)...), numBuffers);
    return stateSlots[slotIndex].get<T>();
  }

  /// Returns the state slot at slotIndex. The state slot must have already
  /// been emplaced.
  StateSlot &getStateSlot(size_t slotIndex) {
    assert(slotIndex < stateSlots.size() && "slotIndex is out of range");
    assert(stateSlots[slotIndex] &&
           "StateSlot for the given slot has not been set or emplaced");
    return stateSlots[slotIndex];
  }

  /// Returns the payload in state slot at slotIndex. The state slot must
  /// have already been emplaced and be of type T.
  template <typename T>
  T &getStateSlotPayload(size_t slotIndex) {
    return getStateSlot(slotIndex).get<T>();
  }

private:
  std::vector<StateSlot> stateSlots;
};

} // namespace M::GML

#endif // GENERICML_SUPPORT_CONTEXT_H
