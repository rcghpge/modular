//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/BufferRef.h"

#include "LLCL/Runtime/Runtime.h"
#include "Support/ML/SizeUtils.h"
#include "llvm/Support/Format.h"

using namespace M;
using namespace M::GML;

static constexpr llvm::MaybeAlign
makeMaybeAlign(std::optional<size_t> alignment) {
  return alignment.has_value() ? llvm::MaybeAlign(*alignment) : std::nullopt;
}

static const char *ownershipToStr(BufferRef::Ownership ownership) {
  switch (ownership) {
  case BufferRef::kNull:
    return "null";
  case BufferRef::kOwned:
    return "owned";
  case BufferRef::kBorrowed:
    return "borrowed";
  }
}

static const char *lifetimeToStr(BufferRef::Lifetime lifetime) {
  switch (lifetime) {
  case BufferRef::kLocal:
    return "local";
  case BufferRef::kExecute:
    return "execute";
  case BufferRef::kConstant:
    return "constant";
  }
}

static const char *mutabilityToStr(BufferRef::Mutability mutability) {
  switch (mutability) {
  case BufferRef::kImmutable:
    return "immutable";
  case BufferRef::kMutable:
    return "mutable";
  }
}

BufferRef BufferRef::createOwned(LLCL::CompactRuntimePtr runtime, size_t size,
                                 std::optional<size_t> alignment) {
  void *buffer = runtime->allocateBytes(size, alignment.value_or(1));
  return BufferRef(buffer, size, runtime, makeMaybeAlign(alignment), kOwned,
                   kLocal, kMutable);
}

BufferRef BufferRef::createBorrowed(void *buffer, size_t size,
                                    std::optional<size_t> alignment,
                                    Lifetime lifetime) {
  assert((!alignment || llvm::isAddrAligned(llvm::Align(*alignment), buffer)) &&
         "buffer is not aligned as per purported alignment");
  return BufferRef(buffer, size, /*runtime=*/nullptr, makeMaybeAlign(alignment),
                   kBorrowed, lifetime, kMutable);
}

BufferRef BufferRef::createBorrowed(const void *buffer, size_t size,
                                    std::optional<size_t> alignment,
                                    Lifetime lifetime) {
  assert((!alignment || llvm::isAddrAligned(llvm::Align(*alignment), buffer)) &&
         "buffer is not aligned as per purported alignment");
  // We make up for the const cast on buffer by setting the reference to
  // be immutable. This will assert fail if a mutable buffer pointer is
  // requested.
  return BufferRef(const_cast<void *>(buffer), size, /*runtime=*/nullptr,
                   makeMaybeAlign(alignment), kBorrowed, lifetime, kImmutable);
}

BufferRef::~BufferRef() {
  if (ownership == kOwned)
    runtime->deallocateBytes(buffer, size);
}

BufferRef::BufferRef(BufferRef &&rhs) { swap(rhs); }

BufferRef &BufferRef::operator=(BufferRef &&rhs) {
  swap(rhs);
  return *this;
}

ErrorOr<BufferRef> BufferRef::borrowOrError(Lifetime borrowLifetime) const {
  assert(getBuffer() != nullptr && "borrowing a null reference");
  if (isOwned() || lifetime >= borrowLifetime)
    return BufferRef(buffer, size, /*runtime=*/nullptr, alignment, kBorrowed,
                     borrowLifetime, mutability);
  return Error(
      Twine("attempting to create borrowing reference with lifetime ") +
      lifetimeToStr(borrowLifetime) +
      " from borrowing reference with lifetime " + lifetimeToStr(lifetime) +
      " which could cause an overhang");
}

BufferRef BufferRef::borrow(Lifetime borrowLifetime) const {
  ErrorOr<BufferRef> errOrBuf = borrowOrError(borrowLifetime);
  assert(!errOrBuf.isError() && "borrow would overhang");
  return std::move(*errOrBuf);
}

BufferRef BufferRef::deepCopy(LLCL::CompactRuntimePtr newRuntime) const {
  assert(getBuffer() != nullptr && "deepCopying a null reference");
  BufferRef ref = createOwned(newRuntime, size, getAlignment());
  std::memcpy(ref.buffer, buffer, size);
  return ref;
}

void BufferRef::printDebug(raw_ostream &os) const {
  os << "BufferRef(";
  os << llvm::format_hex(reinterpret_cast<intptr_t>(buffer),
                         2 + 2 * sizeof(intptr_t));
  os << ", ";
  os << size;
  os << ", ";
  os << ownershipToStr(ownership);
  os << ", ";
  os << lifetimeToStr(lifetime);
  os << ", ";
  os << mutabilityToStr(mutability);
  os << ")";
}

void BufferRef::swap(BufferRef &rhs) {
  std::swap(buffer, rhs.buffer);
  std::swap(size, rhs.size);
  std::swap(runtime, rhs.runtime);
  std::swap(alignment, rhs.alignment);
  std::swap(ownership, rhs.ownership);
  std::swap(lifetime, rhs.lifetime);
  std::swap(mutability, rhs.mutability);
}
