//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ArraySupport/StateSlot.h"

#include "ArraySupport/BufferRef.h"

using namespace M;
using namespace M::GML;

BufferRef StateSlot::getOrCreateBuffer(size_t bufferIndex,
                                       LLCL::CompactRuntimePtr runtime,
                                       size_t size,
                                       std::optional<size_t> alignment) {
  assert(bufferIndex < bufferCache.size() &&
         "buffer index out of bounds in StateSlot");
  auto &entry = bufferCache[bufferIndex];
  updateCacheIfNeeded(entry, runtime, size, alignment);
  // Return a borrowed reference to the cached buffer.
  return entry.borrow();
}

ErrorOr<BufferRef> StateSlot::getBuffer(size_t bufferIndex) {
  assert(bufferIndex < bufferCache.size() &&
         "buffer index out of bounds in StateSlot");
  auto &entry = bufferCache[bufferIndex];
  if (entry.getBuffer() == nullptr)
    return Error("no buffer has been cached at the given index");
  // Return a borrowed reference to the cached buffer. Since it's in the
  // cache, and the compiler follows the 'one shot' rules for reallocating
  // buffers, we can give the borrow 'execute' lifetime.
  return entry.borrow();
}

ErrorOrSuccess StateSlot::setBuffer(size_t bufferIndex, BufferRef bufferRef) {
  assert(bufferIndex < bufferCache.size() &&
         "buffer index out of bounds in StateSlot");
  auto &entry = bufferCache[bufferIndex];
  if (entry.getBuffer() != nullptr)
    return Error("buffer has already been cached at the given index");
  entry = std::move(bufferRef);
  return success();
}

void StateSlot::updateCacheIfNeeded(BufferRef &entry,
                                    LLCL::CompactRuntimePtr runtime,
                                    size_t size,
                                    std::optional<size_t> alignment) {
  // Do nothing if size matches. The size will be 0 if the entry is
  // uninitialized, thus forcing an allocation.
  if (entry.getSize() == size)
    return;

  // Free the existing buffer if exists and create a new one.
  // Since the compiler follows the 'one shot' rules for reallocating
  // buffers, this cannot invalidate any existing borrows.
  entry = BufferRef::createOwned(runtime, size, alignment);
}
