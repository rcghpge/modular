//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/BufferRef.h"

#include "GenericML/Support/Assert.h"
#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/ML/SizeUtils.h"
#include "llvm/Support/Format.h"

using namespace M;
using namespace M::GML;

// We'll need an address for zero sized buffers. Use this as the target.
static constexpr uint64_t kDummyTargetOfEmptyBuffer = 0xdeadbeef;

static void *getDummyTargetOfEmptyBuffer() {
  return const_cast<void *>(
      reinterpret_cast<const void *>(&kDummyTargetOfEmptyBuffer));
}

static constexpr llvm::MaybeAlign
makeMaybeAlign(std::optional<size_t> alignment) {
  return alignment.has_value() ? llvm::MaybeAlign(*alignment) : std::nullopt;
}

BufferRef BufferRef::create(void *buffer, size_t size,
                            AnyAsyncValueRef storageHandle,
                            std::optional<size_t> alignment) {
  return BufferRef::create(buffer, size, std::move(storageHandle),
                           makeMaybeAlign(alignment));
}

BufferRef BufferRef::create(void *buffer, size_t size,
                            AnyAsyncValueRef storageHandle,
                            llvm::MaybeAlign alignment) {
  auto profilerEntry = BufferRefLifetimeProfilerEntry::create(
      "mem.buffer.create", (uint64_t)size);
  return BufferRef(buffer, size, storageHandle.copy(), alignment,
                   std::move(profilerEntry));
}

BufferRef
BufferRef::createWithNonTrackedMemory(void *buffer, size_t size,
                                      std::optional<size_t> alignment) {
  auto profilerEntry = BufferRefLifetimeProfilerEntry::create(
      "mem.buffer.create.nontracked", (uint64_t)size);
  return BufferRef(buffer, size, /*ownedRef=*/{}, makeMaybeAlign(alignment),
                   std::move(profilerEntry));
}

BufferRef BufferRef::createOwned(LLCL::CompactRuntimePtr runtime, size_t size,
                                 std::optional<size_t> alignment,
                                 const void *copyFrom) {
  // Special case for empty buffer
  if (size == 0) {
    return BufferRef::createWithNonTrackedMemory(getDummyTargetOfEmptyBuffer(),
                                                 size, alignment);
  }

  AnyAsyncValueRef owningRef;
  owningRef = owningRef.createReady<OwnedMemoryBlock>(runtime, size, alignment,
                                                      runtime);
  void *buffer = owningRef.get<OwnedMemoryBlock>().data;
  if (copyFrom) {
    LLCL::profiledMemcpy(buffer, copyFrom, size);
  } else {
#if MODULAR_PARANOID
    // Set the buffer contents to a value which will more obviously
    // upset tests if used before initialized.
    memset(buffer, 0xad, size);
#endif
  }
  return BufferRef::create(buffer, size, std::move(owningRef), alignment);
}

BufferRef::~BufferRef() {
#if MODULAR_PARANOID
  // This buffer should have no active uses.
  if (resource)
    resource->markFreed();
#endif
}

BufferRef::BufferRef(BufferRef &&rhs) { swap(rhs); }

BufferRef &BufferRef::operator=(BufferRef &&rhs) {
  swap(rhs);
  return *this;
}

BufferRef BufferRef::borrow() const {
  return BufferRef::create(
      data, size, owningRef ? owningRef.copy() : AnyAsyncValueRef{}, alignment);
}

BufferRef BufferRef::deepCopy(LLCL::CompactRuntimePtr newRuntime) const {
  return createOwned(newRuntime, getSize(), getAlignment(), getBuffer());
}

void BufferRef::reset() {
  data = nullptr;
  size = 0;
  // Though the underlying buffer has not been freed, stop the profiling
  // entry since we can no longer track its lifetime.
  std::move(profilerEntry).record();
#if MODULAR_PARANOID
  // Since the lifetime of the underlying buffer can no longer be tracked,
  // we must consider it freed from the pov of any other borrows.
  if (resource) {
    resource->markFreed();
    resource.reset();
  }
  use.reset();
#endif
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
  os << ")";
}

void BufferRef::swap(BufferRef &rhs) {
  std::swap(data, rhs.data);
  std::swap(size, rhs.size);
  std::swap(owningRef, rhs.owningRef);
  std::swap(alignment, rhs.alignment);
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

OwnedMemoryBlock::OwnedMemoryBlock(size_t size, std::optional<size_t> alignment,
                                   LLCL::CompactRuntimePtr rt)
    : data(nullptr), size(size), runtime(rt) {
  data = runtime->getAllocator()->allocateBytes(size, alignment.value_or(1));
}

OwnedMemoryBlock::~OwnedMemoryBlock() {
  runtime->getAllocator()->deallocateBytes(data, size);
}
