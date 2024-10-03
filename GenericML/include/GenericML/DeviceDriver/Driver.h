//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_LIB_DEVICEDRIVER_DRIVER
#define GENERICML_LIB_DEVICEDRIVER_DRIVER

#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "GenericML/API/max/c/symbol_export.h"
#include "GenericML/DeviceDriver/Utils.h"
#include "Support/DeviceSpecs.h"
#include "Support/ErrorOr.h"
#include "Support/ML/TensorSpec.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "llvm/ADT/FunctionExtras.h"
#include <string>
#include <utility>
namespace M {

namespace Driver {
// Maps compile time device info to runtime device handle

struct Device;
struct CPUDeviceMemory;

using DeviceMap = llvm::DenseMap<DeviceRef, RCRef<M::Driver::Device>>;

/// Kinds of devices that are supported.
enum DeviceKind {
  K_CPU,
  K_CUDA,
};

struct DeviceMemory {
  /// Constructs and allocates DeviceMemory on `dev`, bumping the refcount
  /// of `dev`.
  DeviceMemory(DeviceKind kind, void *allocatedPtr, const TensorSpec &spec,
               Device *dev)
      : devicePtr(M::copyRCRef(dev)), ptr(allocatedPtr), spec(spec),
        kind(kind) {}

  /// Relinquishes ownership of the underlying data, and returns the pointer.
  void *release() { return std::exchange(ptr, nullptr); }

  void setDestructorFn(
      llvm::unique_function<void(void *, const TensorSpec &spec)> destructor) {
    destructorFn = std::move(destructor);
  }

  /// Returns the ptr backing the DeviceMemory object. Does not release
  /// ownership.
  void *data() const { return ptr; }

  size_t bytecount() const { return spec.getSizeInBytes(); }

  virtual ~DeviceMemory();

  DeviceKind getKind() const { return kind; }

  const TensorSpec &getSpec() const { return spec; }

  TensorSpec getSpec() { return spec; }

  M::RCRef<Device> devicePtr;

  /// Copies the contents of DeviceMemory to another chunk of DeviceMemory,
  /// which may reside on a different device.
  virtual ErrorOrSuccess copyTo(DeviceMemory &dst) const = 0;

  /// Sets the contents of the device memory to the provided value.
  virtual ErrorOrSuccess memset(int value) = 0;

protected:
  void *ptr;
  M::TensorSpec spec;
  const DeviceKind kind;
  std::optional<llvm::unique_function<void(void *, const TensorSpec &spec)>>
      destructorFn = std::nullopt;
};

struct Device : M::ReferenceCounted<Device> {

  Device(DeviceSpec spec) : deviceSpecification(std::move(spec)) {}

  virtual const std::string &getString() const { return deviceDescription; };

  /// Allocates `bytes` bytes on the device.
  virtual ErrorOr<void *> allocateData(size_t bytes) = 0;

  /// Frees ptr created with `allocateData`.
  virtual ErrorOrSuccess freeData(void *ptr) = 0;

  DeviceSpec getDeviceSpec() const { return deviceSpecification; };

  /// Allocates `bytes` bytes of memory on the device
  /// and returns the allocation in a DeviceMemory object.
  /// DeviceMemory contains a RCRef of the underlying Device.
  virtual ErrorOr<DeviceMemory *>
  createDeviceMemory(const TensorSpec &spec) = 0;

  /// Takes ownership of `data` in a DeviceMemory object.
  /// DeviceMemory contains a RCRef of the underlying Device.
  virtual DeviceMemory *createDeviceMemory(void *data,
                                           const TensorSpec &spec) = 0;

  /// Takes ownership of `data` in a DeviceMemory object.
  /// DeviceMemory contains a RCRef of the underlying Device. The data
  /// will be destroyed by the given destructor
  virtual DeviceMemory *createDeviceMemory(
      void *data, const TensorSpec &spec,
      llvm::unique_function<void(void *, const TensorSpec &)> destructor) {
    auto deviceMemory = createDeviceMemory(data, spec);
    deviceMemory->setDestructorFn(std::move(destructor));
    return deviceMemory;
  }

  virtual ~Device() = default;

  /// TODO(): This is temporarily needed to unwrap the
  /// DeviceContext pointer from the CUDA Device impl,
  /// because the graph compiler expects it. In the future,
  /// the graph compiler should expect a Device* and this
  /// should be removed.
  virtual void *tmpGetDeviceContext() = 0;

  /// Blocks until any pending async work on the device completes.
  virtual ErrorOrSuccess synchronize() = 0;

  virtual DeviceKind getKind() const = 0;

  /// Gets utilization stats for the device.
  virtual ErrorOr<std::string> getStats() = 0;

  DeviceSpec deviceSpecification;
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

MODULAR_API_EXPORT M::Driver::Device *M_copyDevice(M::Driver::Device *devPtr);

MODULAR_API_EXPORT void M_freeDeviceData(M::Driver::Device *ptr, void *data,
                                         M::Driver::Status *status);

MODULAR_API_EXPORT void M_destroyDevice(M::Driver::Device *devPtr);

MODULAR_API_EXPORT const char *M_getDeviceDesc(const M::Driver::Device *devPtr);

#endif
