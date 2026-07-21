//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// API-layer stub for the HAL DeviceContext replacement header
// (HALDeviceContext.h).
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/HALDeviceContext.h"

#include "Support/ErrorOr.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/Twine.h"

#include <cstdlib>
#include <mutex>

namespace M {
namespace Driver {

// ===-- Device ----------------------------------------------------------=== //

int Device::numberOfDevices(StringRef api) { std::abort(); }

ErrorOr<DeviceRef> Device::create(StringRef api, int id) { std::abort(); }

ErrorOr<DeviceContextRef> Device::createContext() { std::abort(); }

Device::~Device() { std::abort(); }

// Also provided by CompilerDeviceArch.cpp; that TU is a separate library linked
// by disjoint binaries (the compiler family), so this abort definition closes
// the link for the HAL runtime without conflicting.
const std::string &Device::getAcceleratorArchOrEmpty() { std::abort(); }

Device::API Device::api() const { std::abort(); }

int Device::id() const { std::abort(); }

const std::string &Device::name() const { std::abort(); }

DLDevice Device::getDLDevice(bool pinned) const { std::abort(); }

ErrorOr<const char *> Device::getArchName() const { std::abort(); }

ErrorOrSuccess Device::isCompatible() const { std::abort(); }

ErrorOr<Device::MemInfo> Device::getMemoryInfo() const { std::abort(); }

ErrorOr<int> Device::getAttribute(Attribute attr) const { std::abort(); }

ErrorOr<size_t> Device::maxSingleAllocationSize() const { std::abort(); }

ErrorOr<bool> Device::canAccess(const Device &other) const { std::abort(); }

// ===-- DeviceBuffer ----------------------------------------------------=== //

DeviceBuffer::~DeviceBuffer() { std::abort(); }

void *DeviceBuffer::ptr() const { std::abort(); }

size_t DeviceBuffer::byteSize() const { std::abort(); }

bool DeviceBuffer::isHost() const { std::abort(); }

DeviceContext &DeviceBuffer::context() { std::abort(); }

const DeviceContext &DeviceBuffer::context() const { std::abort(); }

void DeviceBuffer::setDestructorFn(llvm::unique_function<void()> destructor) {
  std::abort();
}

ErrorOr<DeviceBufferRef> DeviceBuffer::createSubBuffer(size_t byteOffset,
                                                       size_t byteSize) {
  std::abort();
}

void *DeviceBuffer::releasePtr() { std::abort(); }

ErrorOrSuccess DeviceBuffer::copyTo(DeviceBuffer &dst, size_t srcByteOffset,
                                    size_t srcByteSize) {
  std::abort();
}

ErrorOrSuccess DeviceBuffer::synchronize(void *external_stream) {
  std::abort();
}

ErrorOrSuccess DeviceBuffer::reassignOwnershipTo(DeviceContext &ctx) {
  std::abort();
}

// ===-- DeviceContext ---------------------------------------------------=== //

ErrorOr<DeviceContextRef> DeviceContext::create(StringRef api, int id) {
  std::abort();
}

ErrorOr<DeviceContextRef> DeviceContext::getOrCreate(StringRef api, int id) {
  std::abort();
}

int64_t DeviceContext::configAsInt(StringRef key, int64_t defaultValue,
                                   bool *isDefault) {
  std::abort();
}

DeviceContext::~DeviceContext() { std::abort(); }

Device &DeviceContext::device() { std::abort(); }

const Device &DeviceContext::device() const { std::abort(); }

ErrorOr<DeviceBufferRef> DeviceContext::enqueueCreateBuffer(size_t byteSize) {
  std::abort();
}

ErrorOr<DeviceBufferRef> DeviceContext::createHostBuffer(size_t byteSize) {
  std::abort();
}

DeviceBufferRef DeviceContext::createBufferOwning(void *ptr, size_t byteSize,
                                                  bool owning) {
  std::abort();
}

DeviceBufferRef
DeviceContext::createHostBufferOwning(void *ptr, size_t byteSize, bool owning) {
  std::abort();
}

ErrorOr<DeviceBufferRef>
DeviceContext::memmapFile(const std::filesystem::path &filename) {
  std::abort();
}

ErrorOrSuccess DeviceContext::enqueueCopyToDevice(void *dst, const void *src,
                                                  size_t byteSize) {
  std::abort();
}

ErrorOrSuccess DeviceContext::enqueueCopyFromDevice(void *dst, const void *src,
                                                    size_t byteSize) {
  std::abort();
}

ErrorOrSuccess DeviceContext::enqueueSetMemory(void *dst, size_t byteSize,
                                               uint64_t value,
                                               size_t valueSize) {
  std::abort();
}

ErrorOrSuccess DeviceContext::synchronize(void *external_stream) {
  std::abort();
}

ErrorOr<Device::MemInfo> DeviceContext::getMemoryInfo() const { std::abort(); }

ErrorOr<const char *> DeviceContext::getArchName() const { std::abort(); }

ErrorOrSuccess DeviceContext::isCompatible() const { std::abort(); }

// ===-- DeviceContextScope ----------------------------------------------=== //

DeviceContextScope::DeviceContextScope(const DeviceContext &ctx) {
  std::abort();
}

DeviceContextScope::~DeviceContextScope() { std::abort(); }

ErrorOrSuccess DeviceContextScope::err() const { std::abort(); }

// ===-- CompletionFlag --------------------------------------------------=== //

CompletionFlag::~CompletionFlag() { std::abort(); }

uint64_t CompletionFlag::devicePtr() const { std::abort(); }

void CompletionFlag::reset() { std::abort(); }

void CompletionFlag::signal(uint64_t value) { std::abort(); }

uint64_t CompletionFlag::load() const { std::abort(); }

// ===-- DeviceStream ----------------------------------------------------=== //

DeviceStream::~DeviceStream() { std::abort(); }

Device &DeviceStream::device() const { std::abort(); }

ErrorOrSuccess DeviceStream::synchronize() { std::abort(); }

ErrorOr<DeviceEventRef> DeviceStream::enqueueEvent() { std::abort(); }

ErrorOrSuccess DeviceStream::enqueueWaitForEvent(DeviceEvent &event) {
  std::abort();
}

ErrorOrSuccess DeviceStream::enqueueWaitOnHostValue(const CompletionFlag &flag,
                                                    uint64_t value) const {
  std::abort();
}

ErrorOrSuccess DeviceStream::enqueueHostFunc(void (*fn)(void *),
                                             void *userData) const {
  std::abort();
}

void *DeviceStream::ptr() const { std::abort(); }

// ===-- DeviceEvent -----------------------------------------------------=== //

DeviceEvent::~DeviceEvent() { std::abort(); }

Device &DeviceEvent::device() const { std::abort(); }

ErrorOrSuccess DeviceEvent::synchronize() { std::abort(); }

ErrorOr<bool> DeviceEvent::isReady() const { std::abort(); }

ErrorOr<float> DeviceEvent::elapsedTime(const DeviceEvent &end) const {
  std::abort();
}

// ===-- DeviceContext streams / events ----------------------------------=== //

ErrorOr<DeviceStreamRef> DeviceContext::stream() const { std::abort(); }

ErrorOr<DeviceStreamRef> DeviceContext::createStream(int priority) {
  std::abort();
}

ErrorOr<DeviceStreamRef> DeviceContext::createExternalStream(void *) {
  std::abort();
}

unsigned DeviceContext::numStreams() const { std::abort(); }

ErrorOr<DeviceContextRef> DeviceContext::selectStream(unsigned streamId) {
  std::abort();
}

bool DeviceContext::operator==(const DeviceContext &other) const {
  std::abort();
}

ErrorOr<DeviceEventRef> DeviceContext::enqueueEvent() { std::abort(); }

ErrorOrSuccess DeviceContext::enqueueEvent(const DeviceEvent &event) {
  std::abort();
}

ErrorOr<DeviceEventRef> DeviceContext::eventCreate(bool enableTiming) {
  std::abort();
}

ErrorOrSuccess DeviceContext::enqueueWaitFor(DeviceEvent &event) {
  std::abort();
}

ErrorOrSuccess DeviceContext::enqueueWaitFor(DeviceContext &ctx) {
  std::abort();
}

ErrorOrSuccess DeviceContext::bindToCurrentThread() const { std::abort(); }

std::optional<intptr_t> DeviceContext::getDLPackStream() const { std::abort(); }

// Real error formatter (identical in spirit to CompilerDeviceArch.cpp) so
// ErrorMacros.h keeps working.
Error DeviceContext::formatError(const char *file, int line, const char *func,
                                 const llvm::Twine &msg) {
  return Error(llvm::Twine(file) + ":" + llvm::Twine(line) + " - " + func +
               " failed: " + msg);
}

ErrorOr<CompletionFlagRef> DeviceContext::allocateCompletionFlag() {
  std::abort();
}

// ===-- Device graphs ----------------------------------------------------===
// //

ErrorOr<DeviceGraphCollection>
DeviceGraph::capture(M::AsyncRT::CPUDevice &cpuDevice,
                     llvm::ArrayRef<DeviceStreamRef> streams,
                     llvm::function_ref<ErrorOrSuccess()> execute) {
  std::abort();
}

ErrorOrSuccess DeviceGraph::replay() const { std::abort(); }

ErrorOrSuccess DeviceGraphCollection::replay() const { std::abort(); }

ErrorOrSuccess DeviceGraphCollection::debugVerifyReplay(
    llvm::function_ref<ErrorOrSuccess()> execute) const {
  std::abort();
}

// ===-- Virtual device mode ----------------------------------------------===
// //

// Virtual device mode flags are the process-global AsyncRT runtime globals
// (AsyncRT/Runtime/Globals/VirtualDeviceGlobals.h), shared with the legacy
// backend; only the CPU codegen target below is Driver-owned.

static std::string &getVirtualCpuTargetImpl() {
  static std::string cpu;
  return cpu;
}
static std::mutex &getVirtualCpuTargetMutex() {
  static std::mutex mutex;
  return mutex;
}

MODULAR_CXX_EXPORT void setVirtualCpuTarget(StringRef cpu) {
  std::lock_guard<std::mutex> lock(getVirtualCpuTargetMutex());
  getVirtualCpuTargetImpl() = cpu.str();
}

MODULAR_CXX_EXPORT std::string getVirtualCpuTarget() {
  std::lock_guard<std::mutex> lock(getVirtualCpuTargetMutex());
  return getVirtualCpuTargetImpl();
}

// ===-- BinaryID ----------------------------------------------------------===
// //

MODULAR_CXX_EXPORT std::string getMojoBindingsBinaryID() { std::abort(); }

} // namespace Driver
} // namespace M
