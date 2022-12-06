//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/BufferRef.h"

#include "LLCL/Runtime/Runtime.h"
#include "Support/ML/SizeUtils.h"

using namespace M;
using namespace M::GML;

constexpr llvm::MaybeAlign makeMaybeAlign(Optional<size_t> alignment) {
  return alignment.has_value() ? llvm::MaybeAlign(*alignment) : llvm::None;
}

/// Deallocates underlying buffer if an owning reference.
BufferRef::~BufferRef() {
  if (storage == kOwned)
    runtime->deallocateBytes(buffer, size);
}

// Move is allowed for buffer references.
BufferRef::BufferRef(BufferRef &&rhs)
    : buffer(std::exchange(rhs.buffer, nullptr)),
      size(std::exchange(rhs.size, 0)),
      runtime(std::exchange(rhs.runtime, nullptr)),
      alignment(std::exchange(rhs.alignment, None)),
      storage(std::exchange(rhs.storage, kUninit)) {}

BufferRef &BufferRef::operator=(BufferRef &&rhs) {
  std::swap(buffer, rhs.buffer);
  std::swap(size, rhs.size);
  std::swap(runtime, rhs.runtime);
  std::swap(alignment, rhs.alignment);
  std::swap(storage, rhs.storage);
  return *this;
}

BufferRef BufferRef::borrow() const {
  return createBorrowed(buffer, size, getAlignment());
}

BufferRef BufferRef::deepCopy(LLCL::CompactRuntimePtr newRuntime) const {
  BufferRef ref = createOwned(newRuntime, size, getAlignment());
  std::memcpy(ref.buffer, buffer, size);
  return ref;
}

BufferRef::BufferRef(LLCL::CompactRuntimePtr runtime, size_t size,
                     Optional<size_t> alignment)
    : size(size), runtime(runtime), alignment(makeMaybeAlign(alignment)),
      storage(kOwned) {
  buffer = runtime->allocateBytes(size, getAlignmentOrOne());
}

BufferRef::BufferRef(void *buffer, size_t size, Optional<size_t> alignment)
    : buffer(buffer), size(size), alignment(makeMaybeAlign(alignment)),
      storage(kBorrowed) {}
