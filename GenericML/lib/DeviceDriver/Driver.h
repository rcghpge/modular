//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_LIB_DEVICEDRIVER_DRIVER
#define GENERICML_LIB_DEVICEDRIVER_DRIVER

#include "Support/ErrorOr.h"
#include "Support/ML/TensorSpec.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "llvm/ADT/FunctionExtras.h"
#include <string>
namespace M {

namespace Driver {

struct Device;
struct CPUDeviceMemory;
struct DeviceMemory {
  enum DeviceMemoryKind {
    K_CPU,
    K_CUDA,
  };

  /// Constructs and allocates DeviceMemory on `dev`, bumping the refcount
  /// of `dev`.
  DeviceMemory(DeviceMemoryKind kind, const TensorSpec &spec, Device *dev)
      : devicePtr(M::copyRCRef(dev)), spec(spec), kind(kind) {}

  /// Relinqishes ownership of the underlying data, and returns the pointer.
  void *release() { return std::exchange(ptr, nullptr); }

  void setDestructorFn(
      llvm::unique_function<void(void *, const TensorSpec &spec)> destructor) {
    destructorFn = std::move(destructor);
  }

  /// Returns the ptr backing the DeviceMemory object. Does not release
  /// ownership.
  void *data() { return ptr; }

  size_t bytecount() const { return spec.getSizeInBytes(); }

  virtual ~DeviceMemory();

  DeviceMemoryKind getKind() const { return kind; }

  const TensorSpec &getSpec() const { return spec; }

  TensorSpec getSpec() { return spec; }

  M::RCRef<Device> devicePtr;

  /// Copies the contents of DeviceMemory to another chunk of DeviceMemory,
  /// which may reside on a different device.
  virtual ErrorOrSuccess copy_to(DeviceMemory &dst) const = 0;

protected:
  void *ptr;
  M::TensorSpec spec;
  const DeviceMemoryKind kind;
  std::optional<llvm::unique_function<void(void *, const TensorSpec &spec)>>
      destructorFn = std::nullopt;
};
struct Device : M::ReferenceCounted<Device> {
  virtual const std::string &getString() const = 0;

  /// Allocates `bytes` bytes on the device.
  virtual void *allocateData(size_t bytes) = 0;

  /// Frees ptr created with `allocateData`.
  virtual void freeData(void *ptr) = 0;

  /// Allocates `bytes` bytes of memory on the device
  /// and returns the allocation in a DeviceMemory object.
  /// DeviceMemory contains a RCRef of the underlying Device.
  virtual DeviceMemory *createDeviceMemory(const TensorSpec &spec) = 0;

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
};

} // namespace Driver
} // namespace M

#endif
