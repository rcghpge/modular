//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_LIB_DEVICEDRIVER_DRIVER_H
#define GENERICML_LIB_DEVICEDRIVER_DRIVER_H

#include "GraphCompiler/DeviceDriver/Utils.h"
#include "MLRT/AsyncRT/ForwardDecls.h"
#include "MLRT/AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/DeviceSpecs.h"
#include "Support/ErrorOr.h"
#include "Support/ML/TensorSpec.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/FunctionExtras.h"
#include <string>
#include <utility>
namespace M {
namespace GraphCompiler {
namespace Driver {
// Maps compile time device info to cpuDevice device handle

struct Device;
struct CPUDeviceMemory;

using DeviceMap = llvm::DenseMap<DeviceRef, RCRef<Device>>;

struct DeviceMemory {
  DeviceMemory(RCRef<Device> device, M::Driver::DeviceBufferRef buf,
               const TensorSpec &spec)
      : device(device), buf(buf), spec(spec) {}

  /// Relinquishes ownership of the underlying data, and returns the pointer.
  void *release() { return buf->releasePtr(); }

  /// Underlying raw buffer on-device. Does not release ownership.
  M::Driver::DeviceBufferRef getBuffer() const {
    return M::Driver::DeviceBufferRef::copy(buf.getPointer());
  }

  /// Returns the ptr backing the DeviceMemory object. Does not release
  /// ownership.
  void *data() const { return buf->ptr(); }

  size_t bytecount() const { return buf->byteSize(); }

  DeviceMemory(const DeviceMemory &) = delete;
  DeviceMemory &operator=(const DeviceMemory &) = delete;
  DeviceMemory(DeviceMemory &&) = delete;
  DeviceMemory &operator=(DeviceMemory &&) = delete;

  bool isHost() const {
    return buf->isHost() ||
           buf->context().device().api() == M::Driver::Device::cpuAPI;
  }

  bool isPinned() const { return buf->isHost(); }

  const TensorSpec &getSpec() const { return spec; }

  TensorSpec getSpec() { return spec; }

  RCRef<Device> getDevice() const { return device; }

  /// Copies the contents of DeviceMemory to another chunk of DeviceMemory,
  /// which may reside on a different device.
  ErrorOrSuccess copyTo(DeviceMemory &dst) const;

  ErrorOr<DeviceMemory *> createViewOn(Device &device) const;

  /// Create a view of this memory with a different TensorSpec and optional
  /// byte offset into the underlying buffer.
  ///
  /// This view is a new DeviceMemory object that is backed by the same
  /// underlying memory.
  ErrorOr<DeviceMemory *> createView(const TensorSpec &spec,
                                     size_t byteOffset = 0) const;

  /// Sets the contents of the device memory to the provided value.
  ErrorOrSuccess memset(int value);

protected:
  RCRef<Device> device;
  M::Driver::DeviceBufferRef buf;
  M::TensorSpec spec;
};

struct Device : M::ReferenceCounted<Device> {

  Device(const DeviceSpec &spec, M::Driver::DeviceContextRef ctx)
      : ctx(ctx), deviceSpecification(spec), deviceDescription(toString()) {}

  llvm::StringRef getString() const { return deviceDescription; }

  const DeviceSpec &getDeviceSpec() const { return deviceSpecification; }

  /// Allocates `bytes` bytes of memory on the device
  /// and returns the allocation in a DeviceMemory object.
  /// DeviceMemory contains a RCRef of the underlying Device.
  ErrorOr<DeviceMemory *> createDeviceMemory(const TensorSpec &spec,
                                             bool hostAlloc = false);

  /// Takes ownership of `data` in a DeviceMemory object.
  /// DeviceMemory contains a RCRef of the underlying Device. The data
  /// will be destroyed by the given destructor
  DeviceMemory *createDeviceMemory(void *data, const TensorSpec &spec,
                                   llvm::unique_function<void()> destructor);

  virtual ~Device() = default;

  /// TODO(): This is temporarily needed to unwrap the
  /// DeviceContext pointer from the CUDA Device impl,
  /// because the graph compiler expects it. In the future,
  /// the graph compiler should expect a Device* and this
  /// should be removed.
  M::Driver::DeviceContextRef tmpGetDeviceContext() { return ctx; }

  /// Returns true if this device can access the `peer` device.
  ErrorOr<bool> canAccess(const Device &peer) const {
    return ctx->device().canAccess(peer.ctx->device());
  }

  /// Returns true if `ptr` is memory already resident on this device, so it
  /// can be borrowed in place (zero-copy) instead of staged via a
  /// host->device copy. Used by the C API tensor-borrow path; see DRIV-220.
  bool ownsDeviceAddress(const void *ptr) const {
    return ctx->ownsDeviceAddress(ptr);
  }

  /// Blocks until any pending async work on the device completes.
  ErrorOrSuccess synchronize() { return ctx->synchronize(); }

  bool isHost() const {
    return ctx->device().api() == M::Driver::Device::cpuAPI;
  }

  DeviceId getId() const { return deviceSpecification.ref.id; }

  /// Gets utilization stats for the device.
  ErrorOr<std::string> getStats();

  /// Largest single contiguous allocation the device supports.
  ErrorOr<size_t> getMaxSingleAllocationSize();

  // Gets the programming API of the device.
  ErrorOr<std::string> getApi();

  // Gets the architecture name of the device.
  ErrorOr<std::string> getArchName();

  // Gets the model name of the device (e.g. "NVIDIA B200", "AMD MI300X").
  ErrorOr<std::string> getModelName();

  /// Returns whether or not the device is compatible with MAX.
  ErrorOrSuccess isCompatible() const;

  /// On CPUs, memory maps the given file. On GPUs, direct transfers the data to
  /// the GPU (TODO: unimplemented)
  ErrorOr<M::Driver::DeviceBufferRef>
  memmapFile(const std::filesystem::path &filename);

protected:
  M::Driver::DeviceContextRef ctx;
  DeviceSpec deviceSpecification;
  std::string toString() const;
  std::string deviceDescription;
};

inline std::string formatTarget(const DeviceSpec &deviceSpec) {
  std::string str;
  llvm::raw_string_ostream oss(str);
  oss << "target_info(";
  oss << "triple=" << deviceSpec.target.triple.str() << ",";
  oss << "arch=" << deviceSpec.target.arch << ",";
  oss << "features=[";
  llvm::interleaveComma(deviceSpec.target.features, oss);
  oss << "])";
  return str;
}

/// Returns an array of AnyAsyncValueRef of DeviceContext* corresponding to the
/// deviceRefs array. i.e. ret[i] = AnyAsyncValueRef(deviceMap[deviceRefs[i]]).
/// Note: currently creates an AnyAsyncValueRef of a nullptr when the DeviceRef
/// is not found in the map, expecting the graph compiler to default initialize
/// the DeviceContext in this case. This will be an error in the future once we
/// can create the DeviceContext from cpp (if it has not been supplied by the
/// mojo impl of the MAX driver).
MODULAR_VISIBILITY_EXPORT ErrorOr<llvm::SmallVector<MLRT::AnyAsyncValueRef>>
mapDeviceRefsToDevices(const ArrayRef<DeviceRef> deviceRefs,
                       const DeviceMap &deviceMap, CPUDevice &cpuDevice);

} // namespace Driver
} // namespace GraphCompiler
} // namespace M

MODULAR_DRIVER_EXPORT M::GraphCompiler::Driver::Device *
M_createAcceleratorDevice(ssize_t gpuID,
                          M::GraphCompiler::Driver::Status *status);

MODULAR_DRIVER_EXPORT M::GraphCompiler::Driver::Device *
M_createNPUDevice(ssize_t npuID, M::GraphCompiler::Driver::Status *status);

MODULAR_DRIVER_EXPORT M::GraphCompiler::Driver::Device *
M_createCPUDevice(ssize_t numaId, M::GraphCompiler::Driver::Status *status);

MODULAR_DRIVER_EXPORT void
M_destroyDevice(M::GraphCompiler::Driver::Device *devPtr);

MODULAR_DRIVER_EXPORT const char *
M_getDeviceDesc(const M::GraphCompiler::Driver::Device *devPtr);

namespace M::GraphCompiler::Driver {

/// Creates a RAII wrapped Device object.
MODULAR_VISIBILITY_EXPORT
ErrorOr<std::unique_ptr<Device, decltype(&M_destroyDevice)>> makeCPUDevice();

}; // namespace M::GraphCompiler::Driver

#endif
