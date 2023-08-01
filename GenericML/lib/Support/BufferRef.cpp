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
  void *buffer;
  if (size) {
    buffer =
        runtime->getAllocator()->allocateBytes(size, alignment.value_or(1));
    assert(buffer && "allocator returned null");
    if (copyFrom) {
      LLCL::profiledMemcpy(buffer, copyFrom, size);
    } else {
#if MODULAR_PARANOID
      // Set the buffer contents to a value which will more obviously
      // upset tests if used before initialized.
      memset(buffer, 0xad, size);
#endif
    }
  } else {
    buffer = (void *)&kDummyTargetOfEmptyBuffer;
  }

  auto profilerEntry =
      BufferRefLifetimeProfilerEntry::create("mem.buffer.owned", [size]() {
        return Twine("size:").concat(Twine(size)).concat("B").str();
      });

  return BufferRef(buffer, size, /*owningRef=*/{}, makeMaybeAlign(alignment),
                   kOwned, kLocal, kMutable, runtime, std::move(profilerEntry)
#if MODULAR_PARANOID
                                                          ,
                   /*resource=*/LLCL::Resource::allocate(
                       Twine("owned(").concat(Twine(size)).concat("B)").str()),
                   /*use=*/{}
#endif
  );
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
  return BufferRef(
      borrowedBuffer, size, std::move(owningRef), makeMaybeAlign(alignment),
      kBorrowed, lifetime, kMutable,
      /*runtime=*/{}, /*profilerEntry=*/{}
#if MODULAR_PARANOID
      ,
      /*resource=*/LLCL::Resource::allocate(
          Twine("borrowed(").concat(Twine(size)).concat("B)").str()),
      /*use=*/{}
#endif
  );
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
  return BufferRef(
      const_cast<void *>(borrowedBuffer), size, std::move(owningRef),
      makeMaybeAlign(alignment), kBorrowed, lifetime, kImmutable,
      /*runtime=*/{}, /*profilerEntry=*/{}
#if MODULAR_PARANOID
      ,
      /*resource=*/LLCL::Resource::allocate(
          Twine("borrowed(").concat(Twine(size)).concat("B)").str()),
      /*use=*/{}
#endif
  );
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

  return BufferRef(
      id, object, size, {} /*std::move(owningRef)*/, makeMaybeAlign(alignment),
      kForeign, kLocal, kImmutable,
      /*profilerEntry=*/{}
#if MODULAR_PARANOID
      ,
      /*resource=*/LLCL::Resource::allocate(
          Twine("foreign(").concat(Twine(size)).concat("B)").str()),
      /*use=*/{}
#endif
  );
}

BufferRef BufferRef::createSafeBorrowedSlice(AsyncValueRef<BufferRef> borrowe,
                                             size_t offset, size_t size) {
  assert(borrowe.isValueAvailable() &&
         "given AsyncValue reference must be ready and not an error");
  BufferRef &borroweBufferRef = borrowe.get();
  assert(borroweBufferRef && "cannot borrow from null buffer reference");
  assert(offset <= borroweBufferRef.size && "given offset must be in range");
  if (size == std::numeric_limits<size_t>::max())
    size = borroweBufferRef.size - offset;
  assert(offset + size <= borroweBufferRef.size &&
         "given size must be in range");

  const void *newBuffer = borroweBufferRef.getBuffer<uint8_t>() + offset;

  // We can preserve the borrowe's alignment if the offset is compatible with
  // it. Otherwise fall back to unknown alignment.
  std::optional<size_t> alignment = borroweBufferRef.getAlignment();
  if (size && alignment &&
      !llvm::isAddrAligned(llvm::Align(*alignment), newBuffer))
    alignment = std::nullopt;

  return BufferRef(
      const_cast<void *>(newBuffer), size, std::move(borrowe),
      makeMaybeAlign(alignment), kBorrowed, kLocal, borroweBufferRef.mutability,
      /*runtime=*/{}, /*profilerEntry=*/{}
#if MODULAR_PARANOID
      ,
      /*resource=*/LLCL::Resource::allocate(
          Twine("borrowed(").concat(Twine(size)).concat("B)").str()),
      /*use=*/
      borroweBufferRef.resource->beginUse(
          "slice", LLCL::kReferencingResourceUse,
          LLCL::ResourceSection(offset, offset + size))
#endif
  );
}

BufferRef::~BufferRef() {
#if MODULAR_PARANOID
  // This buffer should have no active uses.
  if (resource)
    resource->markFreed();
#endif
  switch (ownership) {
  case kNull:
    break;
  case kOwned: {
    if (size) {
#if MODULAR_PARANOID
      // Set the buffer to a known value which will more obviously upset
      // unit tests if used again.
      memset(bufferOrObject, 0xfe, size);
#endif
      nativeOrForeign.runtime->getAllocator()->deallocateBytes(bufferOrObject,
                                                               size);
    }
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
      lifetime >= borrowLifetime) {
    return BufferRef(
        const_cast<void *>(getBuffer()), size, owningRef.copy(), alignment,
        kBorrowed, borrowLifetime, mutability, /*runtime=*/{},
        /*profilerEntry=*/{}
#if MODULAR_PARANOID
        ,
        /*resource=*/LLCL::Resource::allocate(
            Twine("borrowed(").concat(Twine(size)).concat("B)").str()),
        /*use=*/
        resource->beginUse("borrow", LLCL::kReferencingResourceUse,
                           LLCL::ResourceSection(0, size))
#endif
    );
  }
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

void BufferRef::reset() {
  ownership = kNull;
  bufferOrObject = nullptr;
  size = 0;
  // Though the underlying buffer has not been freed, stop the profiling
  // entry since we can no longer track its lifetime.
  std::move(profilerEntry).record();
#if MODULAR_DEBUG
  // Since the lifetime of the underlying buffer can no longer be tracked,
  // we must consider it freed from the pov of any other borrows.
  if (resource) {
    resource->markFreed();
    resource.reset();
  }
  use.reset();
#endif
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
#if MODULAR_PARANOID
  std::swap(resource, rhs.resource);
  std::swap(use, rhs.use);
#endif
}

#if MODULAR_PARANOID
void BufferRef::collectAccessUses(SmallVectorImpl<LLCL::ResourceUse> &uses,
                                  StringRef name, LLCL::ResourceUseType useType,
                                  const LLCL::ResourceSection &section) const {
  if (!resource)
    return;
  // We are using this reference.
  uses.emplace_back(resource->beginUse(name.str(), useType, section));
  if (owningRef && owningRef.isType<BufferRef>()) {
    // By transitivity, we are using (a section of) the owningRef's buffer.
    // Account for any (offset, size), which will have been captured by our
    // use.
    const BufferRef &owningBufferRef = owningRef.get<BufferRef>();
    LLCL::ResourceSection parentSection =
        LLCL::isAllResourceSection(section)
            ? use.getSection()
            : LLCL::ResourceSection(section.start() + use.getSection().start(),
                                    section.end() + use.getSection().start());
    owningBufferRef.collectAccessUses(uses, name, useType, parentSection);
  }
}
#endif
