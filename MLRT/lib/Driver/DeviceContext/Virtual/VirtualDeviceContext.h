//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Virtual device context for compile-only mode.
//
// VirtualDeviceContext is a minimal DeviceContext implementation that can be
// used for model compilation without actual hardware. It implements the
// DeviceContext API but most operations are no-ops or return errors if called.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_VIRTUAL_VIRTUALDEVICECONTEXT_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_VIRTUAL_VIRTUALDEVICECONTEXT_H

#include "MLRT/Driver/DeviceContext/CompilationDevice.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

namespace M::Driver {

class VirtualDevice;
class VirtualDeviceContext;
class VirtualDeviceStream;

using VirtualDeviceRef = RCRef<const VirtualDevice>;
using VirtualDeviceContextRef = RCRef<const VirtualDeviceContext>;
using VirtualDeviceStreamRef = RCRef<const VirtualDeviceStream>;

/// Virtual Device for compile-only mode.
///
/// This device mimics GPU devices but doesn't require actual hardware.
/// It's used during compilation to validate graph structure and generate
/// code without needing to execute on real GPUs.
/// Virtual devices use a CompilationDevice for compilation operations.
class VirtualDevice : public Device {
public:
  VirtualDevice(int id, std::string name,
                std::unique_ptr<CompilationDevice> compilationDevice);
  ~VirtualDevice() override;

  ErrorOr<int> getApiVersion() const override;
  ErrorOr<DriverVersion &> getDriverVersion() const override;
  ErrorOrSuccess isCompatible() const override;
  ErrorOr<int> computeCapability() const override;
  ErrorOr<const char *> getArchName() const override;

  ErrorOr<bool> canAccess(const Device &peer) const override;

  ErrorOrSuccess enablePeerAccess(const Device &peer) const override;

  ErrorOr<int> getAttribute(Attribute attr) const override;

  ErrorOr<std::string> getPciBusId() const override;

  int deviceCount() const override;

  ErrorOrSuccess chunkAlloc(void **ptr, size_t byteSize,
                            bool isHost) const override;
  ErrorOrSuccess chunkFree(void *ptr, bool isHost) const override;

  ErrorOr<Device::MemInfo> getMemoryInfo() const override;

  ErrorOr<void *> getCurrentDriverContext() const override;
  ErrorOrSuccess setCurrentDriverContext(void *ctx) const override;

  ErrorOr<DeviceContextRef> createContext() const override;

  static const VirtualDevice &cast(const Device &device);

private:
  // Compilation device used for code generation (e.g., PTX→CUBIN)
  std::unique_ptr<CompilationDevice> compilationDevice_;
};

/// Virtual Device Stream for compile-only mode.
class VirtualDeviceStream : public DeviceStream {
public:
  explicit VirtualDeviceStream(const VirtualDeviceContext &ctx);
  ~VirtualDeviceStream() override = default;

  ErrorOrSuccess synchronize() const override;

  ErrorOr<DeviceEventRef> enqueueEvent() const override;

  ErrorOrSuccess eventRecord(const DeviceEvent &event) const override;
  ErrorOrSuccess waitEvent(const DeviceEvent &event,
                           unsigned int flags) const override;

  ErrorOrSuccess enqueueWaitForEvent(const DeviceEvent &event) const override;

  ErrorOrSuccess enqueueFunctionExecDirect(const DeviceFunction &func,
                                           uint32_t gridX, uint32_t gridY,
                                           uint32_t gridZ, uint32_t blockX,
                                           uint32_t blockY, uint32_t blockZ,
                                           uint32_t sharedMemBytes, void *attrs,
                                           uint32_t numAttrs,
                                           void **args) const override;

  void *ptr() const override;

  static const VirtualDeviceStream *verified_cast(const DeviceStream *stream);
};

/// Virtual Device Context for compile-only mode.
///
/// This context provides minimal functionality needed for compilation:
/// - Memory allocation tracking (doesn't actually allocate)
/// - Stream/event creation (returns stub objects)
/// - Function compilation (uses CompilationDevice)
/// - Synchronization (no-ops)
///
/// Compilation operations use the VirtualDevice's CompilationDevice,
/// which can compile code without requiring GPU hardware.
class VirtualDeviceContext : public DeviceContext {
public:
  explicit VirtualDeviceContext(const VirtualDevice &device);
  ~VirtualDeviceContext() override;

  // DeviceBuffer copies - not implemented for virtual context
  ErrorOrSuccess enqueueCopyToDevice(void *dst, const void *src,
                                     size_t byteSize) const override;
  ErrorOrSuccess enqueueCopyFromDevice(void *dst, const void *src,
                                       size_t byteSize) const override;
  ErrorOrSuccess memCopyDtoD(const DeviceBuffer &dst, const DeviceBuffer &src,
                             size_t srcByteOffset,
                             size_t srcByteSize) const override;

  ErrorOrSuccess enqueueSetMemory(void *dst, size_t byteSize, uint64_t val,
                                  size_t valSize) const override;

  // DeviceFunction compiling - minimal implementation
  ErrorOr<DeviceFunctionRef>
  loadFunction(StringRef moduleName, StringRef functionName, StringRef code,
               int32_t maxDynamicSharedBytes, StringRef debugLevel,
               int32_t optimizationLevel) const override;

  ErrorOrSuccess enqueueFunctionExecDirect(
      const DeviceFunction &func, uint32_t gridX, uint32_t gridY,
      uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
      uint32_t sharedMemBytes, void *attrs, uint32_t numAttrs, void **args,
      const size_t *argSizes = nullptr) const override;

  // Synchronize asynchronous operations - no-ops
  ErrorOr<DeviceStreamRef> stream() const override;
  ErrorOr<DeviceStreamRef> createStream(int priority) const override;
  ErrorOrSuccess streamPriorityRange(int *leastPriority,
                                     int *greatestPriority) const override;
  ErrorOr<DeviceStreamRef>
  createExternalStream(void *externalStream) const override;

  ErrorOrSuccess synchronize() const override;

  ErrorOr<DeviceEventRef> enqueueEvent() const override;
  ErrorOrSuccess enqueueEvent(const DeviceEvent &event) const override;
  ErrorOr<DeviceEventRef> eventCreate(unsigned int flags) const override;
  ErrorOrSuccess enqueueWaitFor(const DeviceEvent &event) const override;
  ErrorOrSuccess enqueueWaitFor(const DeviceContext &ctx) const override;

  ErrorOr<DeviceTimerRef> startTimer() const override;
  ErrorOr<std::chrono::nanoseconds>
  stopTimer(const DeviceTimer &timer) const override;

  ErrorOr<Device::MemInfo> getDeviceMemoryInfo() const override;

  ErrorOrSuccess occupancyMaxActiveBlocksPerMultiprocessor(
      int *numBlocks, const DeviceFunction &func, int blockSize,
      size_t dynamicSharedMemSize) const override;

  static const VirtualDeviceContext &cast(const DeviceContext &ctx);
  static const VirtualDeviceContext *verified_cast(const DeviceContext *ctx);

private:
  ErrorOrSuccess memAlloc(void **ptr, size_t byteSize,
                          Device::AllocMode mode) const override;
  ErrorOrSuccess memFree(void *ptr, bool isHost) const override;

  void *driverContext() const override;

  MMapFileImplResult
  memmapFileImpl(const std::filesystem::path &filename) const override;
};

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_VIRTUAL_VIRTUALDEVICECONTEXT_H
