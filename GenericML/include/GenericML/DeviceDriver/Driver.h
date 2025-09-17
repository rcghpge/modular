//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_LIB_DEVICEDRIVER_DRIVER
#define GENERICML_LIB_DEVICEDRIVER_DRIVER

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "GenericML/DeviceDriver/Utils.h"
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

namespace Driver {
// Maps compile time device info to runtime device handle

struct Device;
struct CPUDeviceMemory;

using DeviceMap = llvm::DenseMap<DeviceRef, RCRef<M::Driver::Device>>;

struct DeviceMemory {
  DeviceMemory(RCRef<Device> device, AsyncRT::DeviceBufferRef buf,
               const TensorSpec &spec)
      : device(device), buf(buf), spec(spec) {}

  /// Relinquishes ownership of the underlying data, and returns the pointer.
  void *release() { return buf->releasePtr(); }

  void setDestructorFn(
      llvm::unique_function<void(void *, const TensorSpec &spec)> destructor) {
    destructorFn = std::move(destructor);
  }

  /// Returns the ptr backing the DeviceMemory object. Does not release
  /// ownership.
  void *data() const { return buf->ptr(); }

  size_t bytecount() const { return buf->byteSize(); }

  ~DeviceMemory();
  DeviceMemory(const DeviceMemory &) = delete;
  DeviceMemory &operator=(const DeviceMemory &) = delete;
  DeviceMemory(DeviceMemory &&) = delete;
  DeviceMemory &operator=(DeviceMemory &&) = delete;

  bool isHost() const {
    return buf->isHost() ||
           buf->context().device().api() == M::AsyncRT::Device::cpuAPI;
  }

  bool isPinned() const { return buf->isHost(); }

  const TensorSpec &getSpec() const { return spec; }

  TensorSpec getSpec() { return spec; }

  RCRef<Device> getDevice() const { return device; }

  /// Copies the contents of DeviceMemory to another chunk of DeviceMemory,
  /// which may reside on a different device.
  ErrorOrSuccess copyTo(DeviceMemory &dst) const;

  ErrorOr<DeviceMemory *> createViewOn(Device &device) const;

  /// Sets the contents of the device memory to the provided value.
  ErrorOrSuccess memset(int value);

protected:
  RCRef<Device> device;
  AsyncRT::DeviceBufferRef buf;
  M::TensorSpec spec;
  std::optional<llvm::unique_function<void(void *, const TensorSpec &spec)>>
      destructorFn = std::nullopt;
};

struct Device : M::ReferenceCounted<Device> {

  Device(const DeviceSpec &spec, AsyncRT::DeviceContextRef ctx)
      : ctx(ctx), deviceSpecification(spec), deviceDescription(toString()) {}

  llvm::StringRef getString() const { return deviceDescription; };

  /// Allocates `bytes` bytes on the device.
  ErrorOr<void *> allocateData(size_t bytes);

  /// Frees ptr created with `allocateData`.
  ErrorOrSuccess freeData(void *ptr);

  const DeviceSpec &getDeviceSpec() const { return deviceSpecification; };

  /// Allocates `bytes` bytes of memory on the device
  /// and returns the allocation in a DeviceMemory object.
  /// DeviceMemory contains a RCRef of the underlying Device.
  ErrorOr<DeviceMemory *> createDeviceMemory(const TensorSpec &spec,
                                             bool hostAlloc = false);

  /// Takes ownership of `data` in a DeviceMemory object.
  /// DeviceMemory contains a RCRef of the underlying Device.
  DeviceMemory *createDeviceMemory(void *data, const TensorSpec &spec);

  /// Takes ownership of `data` in a DeviceMemory object.
  /// DeviceMemory contains a RCRef of the underlying Device. The data
  /// will be destroyed by the given destructor
  DeviceMemory *createDeviceMemory(
      void *data, const TensorSpec &spec,
      llvm::unique_function<void(void *, const TensorSpec &)> destructor);

  virtual ~Device() = default;

  /// TODO(): This is temporarily needed to unwrap the
  /// DeviceContext pointer from the CUDA Device impl,
  /// because the graph compiler expects it. In the future,
  /// the graph compiler should expect a Device* and this
  /// should be removed.
  AsyncRT::DeviceContextRef tmpGetDeviceContext() { return ctx; }

  /// Returns true if this device can access the `peer` device.
  ErrorOr<bool> canAccess(const Device &peer) const {
    return ctx->canAccess(*peer.ctx);
  };

  /// Blocks until any pending async work on the device completes.
  ErrorOrSuccess synchronize() { return ctx->synchronize(); };

  bool isHost() const {
    return ctx->device().api() == M::AsyncRT::Device::cpuAPI;
  }

  DeviceId getId() const { return deviceSpecification.ref.id; }

  /// Gets utilization stats for the device.
  ErrorOr<std::string> getStats();

  // Gets the programming API of the device.
  ErrorOr<std::string> getApi();

  // Gets the architecture name of the device.
  ErrorOr<std::string> getArchName();

  /// Returns whether or not the device is compatible with MAX.
  ErrorOrSuccess isCompatible() const;

  /// On CPUs, memory maps the given file, returns a handle to the memmap'ed
  /// file. On GPUs, direct transfers the data to the GPU (TODO: unimplemented)
  ErrorOr<std::pair<AsyncRT::DeviceBufferRef,
                    std::optional<llvm::sys::fs::mapped_file_region>>>
  memmapFile(const std::filesystem::path &filename);

protected:
  AsyncRT::DeviceContextRef ctx;
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
MODULAR_VISIBILITY_EXPORT ErrorOr<llvm::SmallVector<AnyAsyncValueRef>>
mapDeviceRefsToDevices(const ArrayRef<DeviceRef> deviceRefs,
                       const Driver::DeviceMap &deviceMap, Runtime &runtime);

} // namespace Driver
} // namespace M

/// `deviceMemoryLimit` is the maximum amount of memory in bytes that can be
// allocated on the device. If it is -1, the device will apply defaults.
MODULAR_DRIVER_EXPORT M::Driver::Device *
M_createAcceleratorDevice(ssize_t gpuID, int64_t deviceMemoryLimit,
                          M::Driver::Status *status);

MODULAR_DRIVER_EXPORT M::Driver::Device *
M_createCPUDevice(ssize_t numaId, M::Driver::Status *status);

MODULAR_DRIVER_EXPORT M::Driver::Device *
M_copyDevice(M::Driver::Device *devPtr);

MODULAR_DRIVER_EXPORT void M_freeDeviceData(M::Driver::Device *ptr, void *data,
                                            M::Driver::Status *status);

MODULAR_DRIVER_EXPORT void M_destroyDevice(M::Driver::Device *devPtr);

MODULAR_DRIVER_EXPORT const char *
M_getDeviceDesc(const M::Driver::Device *devPtr);

// Returns number of accelerators available.
MODULAR_DRIVER_EXPORT int M_getAcceleratorCount();

namespace M::Driver {

/// Creates a RAII wrapped Device object.
MODULAR_VISIBILITY_EXPORT
ErrorOr<std::unique_ptr<Device, decltype(&M_destroyDevice)>> makeCPUDevice();

}; // namespace M::Driver

#endif
