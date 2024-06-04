//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ArraySupport/TensorBufferRef.h"

#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Runtime/Runtime.h"
#include "llvm/Support/Format.h"

using namespace M;

static constexpr llvm::MaybeAlign
makeMaybeAlign(std::optional<size_t> alignment) {
  return alignment.has_value() ? llvm::MaybeAlign(*alignment) : std::nullopt;
}

TensorBufferRef TensorBufferRef::create(void *buffer, size_t size,
                                        AnyAsyncValueRef storageHandle,
                                        std::optional<size_t> alignment) {
  return TensorBufferRef::create(buffer, size, std::move(storageHandle),
                                 makeMaybeAlign(alignment));
}

TensorBufferRef TensorBufferRef::create(void *buffer, size_t size,
                                        AnyAsyncValueRef storageHandle,
                                        llvm::MaybeAlign alignment) {
  auto profilerEntry = BufferRefLifetimeProfilerEntry::create(
      "mem.buffer.create", (uint64_t)size);
  return TensorBufferRef(buffer, size, storageHandle.copy(), alignment,
                         std::move(profilerEntry));
}

TensorBufferRef
TensorBufferRef::createWithNonTrackedMemory(void *buffer, size_t size,
                                            std::optional<size_t> alignment) {
  auto profilerEntry = BufferRefLifetimeProfilerEntry::create(
      "mem.buffer.create.nontracked", (uint64_t)size);
  return TensorBufferRef(buffer, size, /*ownedRef=*/{},
                         makeMaybeAlign(alignment), std::move(profilerEntry));
}

TensorBufferRef TensorBufferRef::createOwned(LLCL::CompactRuntimePtr runtime,
                                             size_t size,
                                             std::optional<size_t> alignment) {
  // Special case for empty buffer
  if (size == 0) {
    return TensorBufferRef::createWithNonTrackedMemory(nullptr, size,
                                                       alignment);
  }

  AnyAsyncValueRef storageRef;
  storageRef = storageRef.createReady<OwnedMemoryBlock>(runtime, size,
                                                        alignment, runtime);
  void *buffer = storageRef.get<OwnedMemoryBlock>().data;
#if MODULAR_PARANOID
  // Set the buffer contents to a value which will more obviously
  // upset tests if used before initialized.
  memset(buffer, 0xad, size);
#endif
  return TensorBufferRef::create(buffer, size, std::move(storageRef),
                                 alignment);
}

// Create an owned reference from a taken void* pointer -- This taken
// pointer will not be freed by anyone else except by the buffer-ref now.
TensorBufferRef TensorBufferRef::take(LLCL::CompactRuntimePtr runtime,
                                      size_t size, void *takeFrom) {
  AnyAsyncValueRef storageRef;
  storageRef = storageRef.createReady<OwnedMemoryBlock>(
      runtime, size, std::nullopt, runtime, takeFrom);
  void *buffer = storageRef.get<OwnedMemoryBlock>().data;
  return TensorBufferRef::create(buffer, size, std::move(storageRef),
                                 std::optional<size_t>());
}

TensorBufferRef::~TensorBufferRef() {
#if MODULAR_PARANOID
  // This buffer should have no active uses.
  if (resource)
    resource->markFreed();
#endif
}

TensorBufferRef::TensorBufferRef(TensorBufferRef &&rhs) { swap(rhs); }

TensorBufferRef &TensorBufferRef::operator=(TensorBufferRef &&rhs) {
  swap(rhs);
  return *this;
}

TensorBufferRef TensorBufferRef::borrow() const {
  return TensorBufferRef::create(
      data, size, storageRef ? storageRef.copy() : AnyAsyncValueRef{},
      std::optional<size_t>());
}

void TensorBufferRef::reset() {
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

void TensorBufferRef::printDebug(raw_ostream &os) const {
  os << "TensorBufferRef(";
  os << llvm::format_hex(reinterpret_cast<intptr_t>(getBuffer()),
                         2 + 2 * sizeof(intptr_t));
  os << ", ";
  os << getSize();
  if (storageRef) {
    os << ", ";
    os << *storageRef.getPointer();
  }
  os << ")";
}

void TensorBufferRef::swap(TensorBufferRef &rhs) {
  std::swap(data, rhs.data);
  std::swap(size, rhs.size);
  std::swap(storageRef, rhs.storageRef);
  std::swap(profilerEntry, rhs.profilerEntry);
#if MODULAR_PARANOID
  std::swap(resource, rhs.resource);
  std::swap(use, rhs.use);
#endif
}

#if MODULAR_PARANOID
void TensorBufferRef::collectAccessUses(
    SmallVectorImpl<LLCL::ResourceUse> &uses, StringRef name,
    LLCL::ResourceUseType useType, const LLCL::ResourceSection &section) const {
  if (!resource)
    return;
  // We are using this reference.
  uses.emplace_back(resource->beginUse(name.str(), useType, section));
  if (storageRef && storageRef.isType<TensorBufferRef>()) {
    // By transitivity, we are using (a section of) the storageRef's buffer.
    // Account for any (offset, size), which will have been captured by our
    // use.
    const TensorBufferRef &owningBufferRef = storageRef.get<TensorBufferRef>();
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
                                   LLCL::CompactRuntimePtr rt, void *takePtr)
    : data(nullptr), size(size), runtime(rt) {
  if (takePtr)
    data = takePtr;
  else
    data = runtime->getAllocator()->allocateBytes(size, alignment.value_or(1));
}

OwnedMemoryBlock::~OwnedMemoryBlock() {
  runtime->getAllocator()->deallocateBytes(data, size);
}
