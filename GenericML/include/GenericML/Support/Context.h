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
/// Typically a state slot corresponds to a unique operation in the graph.
/// However state slots can also be used to hold other framework-specific
/// state, such as a framework specific context.
///
/// State slots also generically support caching of buffer allocations.
///
/// No thread synchronization is enforced. A slot index should be emplaced at
/// most once. A state slot's contents should be accessed by one thread at a
/// time. Both these conditions are easy to satisfy if state slots are unique
/// for an operation in the overall graph.
class Context {
public:
  /// Constructs a new context with numSlots uninitialized state slots.
  explicit Context(size_t numSlots) : stateSlots(numSlots) {}
  ~Context() = default;

  /// Returns the number of state slots in the context.
  size_t getNumStateSlots() const { return stateSlots.size(); }

  /// Returns true if the state slot at slotIndex has been emplaced.
  bool hasStateSlot(size_t slotIndex) {
    assert(slotIndex < stateSlots.size() && "slotIndex is out of range");
    return stateSlots[slotIndex] != nullptr;
  }

  // No implicit copying is allowed.
  Context(const Context &rhs) = delete;
  Context &operator=(const Context &rhs) = delete;

  /// Constructs a T in-place within the state slot at slotIndex. The state
  /// slot must not have already been emplaced. The state slot will also hold
  /// space for numBuffers cached buffer allocations, which will all initially
  /// be 'uninitialized'.
  template <typename T, typename... Args>
  T &emplaceStateSlot(size_t slotIndex, size_t numBuffers, Args &&...args) {
    assert(stateSlots[slotIndex] == nullptr &&
           "StateSlot for the given slot already exists.");
    stateSlots[slotIndex] =
        StateSlot::createStateSlot<T>(numBuffers, std::forward<Args>(args)...);
    return getStateSlotPayload<T>(slotIndex);
  }

  /// Returns the state slot at slotIndex. The state slot must have already
  /// been emplaced.
  StateSlot &getStateSlot(size_t slotIndex) const;

  /// Returns the payload in state slot at slotIndex. The state slot must
  /// have already been emplaced and be of type T.
  template <typename T>
  T &getStateSlotPayload(size_t slotIndex) const {
    return getStateSlot(slotIndex).template get<T>();
  }

private:
  std::vector<std::unique_ptr<StateSlot>> stateSlots;
};

} // namespace M::GML

#endif // GENERICML_SUPPORT_CONTEXT_H
