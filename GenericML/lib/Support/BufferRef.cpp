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

// We'll need an address for zero sized buffers. Use this as the target.
static const uint64_t kDummyTargetOfEmptyBuffer = 0xdeadbeef;

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
  case BufferRef::kForeign:
    return "foreign";
  }
  llvm_unreachable("missing ownership case");
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
  llvm_unreachable("missing lifetime case");
}

static const char *mutabilityToStr(BufferRef::Mutability mutability) {
  switch (mutability) {
  case BufferRef::kImmutable:
    return "immutable";
  case BufferRef::kMutable:
    return "mutable";
  }
  llvm_unreachable("missing mutability case");
}

BufferRef BufferRef::createOwned(LLCL::CompactRuntimePtr runtime, size_t size,
                                 std::optional<size_t> alignment,
                                 const void *copyFrom) {
  void *buffer =
      size == 0
          ? (void *)&kDummyTargetOfEmptyBuffer
          : runtime->getAllocator()->allocateBytes(size, alignment.value_or(1));
  if (copyFrom)
    LLCL::profiledMemcpy(buffer, copyFrom, size);
  return BufferRef(buffer, size, /*owningRef=*/{}, makeMaybeAlign(alignment),
                   kOwned, kLocal, kMutable, runtime);
}

BufferRef BufferRef::createBorrowed(void *buffer, size_t size,
                                    std::optional<size_t> alignment,
                                    Lifetime lifetime,
                                    LLCL::AnyAsyncValueRef owningRef) {
  assert((size == 0 || buffer) && "borrowed buffer must be non-null");
  assert((size == 0 || !alignment ||
          llvm::isAddrAligned(llvm::Align(*alignment), buffer)) &&
         "buffer is not aligned as per purported alignment");
  void *borrowedBuffer = size == 0 && buffer == nullptr
                             ? (void *)&kDummyTargetOfEmptyBuffer
                             : buffer;
  return BufferRef(borrowedBuffer, size, std::move(owningRef),
                   makeMaybeAlign(alignment), kBorrowed, lifetime, kMutable,
                   /*runtime=*/{});
}

BufferRef BufferRef::createBorrowed(const void *buffer, size_t size,
                                    std::optional<size_t> alignment,
                                    Lifetime lifetime,
                                    AnyAsyncValueRef owningRef) {
  assert((size == 0 || buffer) && "borrowed buffer must be non-null");
  assert((size == 0 || !alignment ||
          llvm::isAddrAligned(llvm::Align(*alignment), buffer)) &&
         "buffer is not aligned as per purported alignment");
  const void *borrowedBuffer = size == 0 && buffer == nullptr
                                   ? (void *)&kDummyTargetOfEmptyBuffer
                                   : buffer;
  // We make up for the const cast on buffer by setting the reference to
  // be immutable. This will assert fail if a mutable buffer pointer is
  // requested.
  return BufferRef(const_cast<void *>(borrowedBuffer), size,
                   std::move(owningRef), makeMaybeAlign(alignment), kBorrowed,
                   lifetime, kImmutable, /*runtime=*/{});
}

BufferRef BufferRef::createForeign(ForeignBufferInterfaceID id, void *object,
                                   size_t size, std::optional<size_t> alignment,
                                   AnyAsyncValueRef owningRef) {
  assert(object && "foreign object cannot be null");
  // Note that even if size is 0 we are obliged to create a non-null
  // BufferRef so as to retain ownership of object.
  if (alignment && size)
    assert(llvm::isAddrAligned(
               llvm::Align(*alignment),
               ForeignBufferInterfaceRegistry::getBuffer(id, object)) &&
           "buffer is not aligned as per purported alignment");
  return BufferRef(id, object, size, {} /*std::move(owningRef)*/,
                   makeMaybeAlign(alignment), kForeign, kLocal, kImmutable);
}

BufferRef BufferRef::createSafeBorrowedSlice(AsyncValueRef<BufferRef> borrowe,
                                             size_t offset, size_t size) {
  assert(borrowe.isValueAvailable() &&
         "given AsyncValue reference must be ready and not an error");
  BufferRef &borroweBufferRef = borrowe.get();
  assert(offset <= borroweBufferRef.size && "given offset must be in range");
  if (size == std::numeric_limits<size_t>::max())
    size = borroweBufferRef.size - offset;
  assert(offset + size <= borroweBufferRef.size &&
         "given size must be in range");

  const void *newBuffer = borroweBufferRef.getBuffer<uint8_t>() + offset;

  // We can preserve the borrowe's alignment if the offset is compatible with
  // it. Otherwise fall back to unknown alignment.
  std::optional<size_t> alignment = borroweBufferRef.getAlignment();
  if (alignment && !llvm::isAddrAligned(llvm::Align(*alignment), newBuffer))
    alignment = std::nullopt;

  return BufferRef(const_cast<void *>(newBuffer), size, std::move(borrowe),
                   makeMaybeAlign(alignment), kBorrowed, kLocal,
                   borroweBufferRef.mutability, /*runtime=*/{});
}

BufferRef::~BufferRef() {
  switch (ownership) {
  case kNull:
    break;
  case kOwned: {
    if (size)
      nativeOrForeign.runtime->getAllocator()->deallocateBytes(bufferOrObject,
                                                               size);
    std::move(profilerEntry).record();
    break;
  }
  case kBorrowed:
    break;
  case kForeign:
    ForeignBufferInterfaceRegistry::deleteObject(nativeOrForeign.id,
                                                 bufferOrObject);
    break;
  }
}

BufferRef::BufferRef(BufferRef &&rhs) { swap(rhs); }

BufferRef &BufferRef::operator=(BufferRef &&rhs) {
  swap(rhs);
  return *this;
}

ErrorOr<BufferRef> BufferRef::borrowOrError(Lifetime borrowLifetime) const {
  if (ownership == kNull)
    return BufferRef();
  if (ownership == kOwned || ownership == kForeign ||
      lifetime >= borrowLifetime)
    return BufferRef(const_cast<void *>(getBuffer()), getSize(),
                     owningRef.copy(), alignment, kBorrowed, borrowLifetime,
                     mutability, /*runtime=*/{});
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
  if (ownership == kNull)
    return {};
  return createOwned(newRuntime, getSize(), getAlignment(), getBuffer());
}

void *BufferRef::releaseMutableBufferIfOwned() {
  assert(mutability == kMutable && "buffer is immutable");
  if (ownership != kOwned)
    return nullptr;
  void *bufferCopy = bufferOrObject;
  reset();
  return bufferCopy;
}

const void *BufferRef::releaseBufferIfOwned() {
  if (ownership != kOwned)
    return nullptr;
  const void *bufferCopy = bufferOrObject;
  reset();
  return bufferCopy;
}

void *
BufferRef::maybeReleaseForeignObject(ForeignBufferInterfaceID expectedID) {
  if (ownership != kForeign)
    return nullptr;
  if (nativeOrForeign.id != expectedID)
    return nullptr;
  void *objectCopy = bufferOrObject;
  reset();
  return objectCopy;
}

void BufferRef::printDebug(raw_ostream &os) const {
  os << "BufferRef(";
  os << llvm::format_hex(reinterpret_cast<intptr_t>(getBuffer()),
                         2 + 2 * sizeof(intptr_t));
  os << ", ";
  os << getSize();
  if (owningRef) {
    os << ", ";
    os << *owningRef.getPointer();
  }
  os << ", ";
  os << ownershipToStr(ownership);
  os << ", ";
  os << lifetimeToStr(lifetime);
  os << ", ";
  os << mutabilityToStr(mutability);
  if (ownership == kForeign) {
    os << ", ";
    os << std::to_string(nativeOrForeign.id);
  }
  os << ")";
}

void BufferRef::swap(BufferRef &rhs) {
  std::swap(bufferOrObject, rhs.bufferOrObject);
  std::swap(size, rhs.size);
  std::swap(owningRef, rhs.owningRef);
  std::swap(alignment, rhs.alignment);
  std::swap(ownership, rhs.ownership);
  std::swap(lifetime, rhs.lifetime);
  std::swap(mutability, rhs.mutability);
  std::swap(nativeOrForeign, rhs.nativeOrForeign);
  std::swap(profilerEntry, rhs.profilerEntry);
}
