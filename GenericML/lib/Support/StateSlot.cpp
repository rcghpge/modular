//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/StateSlot.h"

#include "GenericML/Support/BufferRef.h"
#include "Support/AlignedAlloc.h"

using namespace M;
using namespace M::GML;

StateSlot::StateSlot(TypeID typeID, size_t numBuffers)
    : typeID(typeID), bufferCache(numBuffers) {
  assert(typeID != kInvalidTypeID);
}

BufferRef StateSlot::getOrCreateBuffer(size_t bufferIndex,
                                       LLCL::CompactRuntimePtr runtime,
                                       size_t size,
                                       Optional<size_t> alignment) {
  assert(bufferIndex < bufferCache.size());
  auto &entry = bufferCache[bufferIndex];
  updateCacheIfNeeded(entry, runtime, size, alignment);
  // Return a borrowed reference to the cached buffer.
  return entry.borrow();
}

ErrorOr<BufferRef> StateSlot::getBuffer(size_t bufferIndex) {
  assert(bufferIndex < bufferCache.size());
  auto &entry = bufferCache[bufferIndex];
  if (entry.getData() == nullptr)
    return Error("no buffer has been cached at the given index");
  // Return a borrowed reference to the cached buffer.
  return entry.borrow();
}

/// Assigns to staticTypeId a globally unique type id.
void StateSlot::doTypeRegistration(std::atomic<TypeID> *staticTypeID) {
  static std::atomic<TypeID> nextFreeTypeId = 0;
  // Allocate the next id. We may end up not using it if another thread
  // beats us to the assignment to staticTypeId below.
  TypeID nextId = nextFreeTypeId++;
  // Detect overflow.
  assert(nextId != kInvalidTypeID &&
         "too many types registered by StateSlot::registerType");
  // Grab the next id. At most one thread will win, we don't care which.
  TypeID existingId = kInvalidTypeID;
  (void)staticTypeID->compare_exchange_strong(
      existingId, nextId, std::memory_order_release, std::memory_order_acquire);
}

void StateSlot::updateCacheIfNeeded(BufferRef &entry,
                                    LLCL::CompactRuntimePtr runtime,
                                    size_t size, Optional<size_t> alignment) {
  // Do nothing if size matches. The size will be 0 if the entry is
  // uninitialized, thus forcing an allocation.
  if (entry.getSize() == size)
    return;

  // Free the existing buffer if exists and create a new one.
  entry = BufferRef::createOwned(runtime, size, alignment);
}
