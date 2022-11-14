//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MOP_CONTEXT_H
#define GENERICML_MOP_CONTEXT_H

#include "GenericML/MOP/PrimContext.h"
#include "Support/AlignedAlloc.h"

namespace M::MOP {

/// This is a global context for Modular framework execution. The global context
/// manages PrimContexts, which is a context local to a particular primitive
/// operation.
class Context {
public:
  /// Construct a new context.
  explicit Context(size_t numSlots) : primContexts(numSlots) {}
  ~Context() = default;

  // No implicit copying is allowed.
  Context(const Context &rhs) = delete;
  Context &operator=(const Context &rhs) = delete;

  /// Create a GenericPrimContext for a specific framework context and register
  /// it to the context with the given slot number.
  template <typename T, typename... Args>
  void emplacePrimContext(size_t slotNumber, size_t numOutputs,
                          Args &&...args) {
    assert(primContexts[slotNumber] == nullptr &&
           "PrimContext for the given slot already exists.");
    primContexts[slotNumber] =
        AnyGenericPrimContext::createGenericPrimContext<T>(
            numOutputs, std::forward<Args>(args)...);
  }

  /// Retrieve the primitive context at given slot number.
  AnyGenericPrimContext &getPrimContext(size_t slotNumber) const {
    auto *ptr = primContexts[slotNumber].get();
    assert(ptr != nullptr);
    return *ptr;
  }

private:
  std::vector<std::unique_ptr<AnyGenericPrimContext>> primContexts;
};

} // namespace M::MOP

#endif
