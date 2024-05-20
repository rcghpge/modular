//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/API/max/c/symbol_export.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/AlignedAlloc.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <sstream>
#include <utility>

namespace M {

namespace Driver {

struct Device : M::ReferenceCounted<Device> {
  virtual const std::string &getString() const = 0;
  virtual void *allocateData(size_t bytes) = 0;
  virtual void freeData(void *ptr) = 0;
  virtual ~Device() = default;
};

struct CPUDevice : Device {
  size_t numa_id;
  std::string desc;

  CPUDevice(size_t numa_id) : numa_id(numa_id), desc(toString()) {}

  ~CPUDevice() override = default;

  void *allocateData(size_t bytes) override {
    return M::alignedAlloc(kPreferredMemoryAlignment, bytes);
  }

  void freeData(void *data) override { alignedFree(data); }

  const std::string &getString() const override { return desc; }

private:
  std::string toString() const {
    std::string str;
    llvm::raw_string_ostream oss(str);
    oss << "Device(type=CPU,numa_id=" << numa_id << ")";
    return str;
  }
};

struct DeviceTensor {
  M::RCRef<Device> devicePtr;
  void *data;
  size_t bytes;
  DeviceTensor(size_t bytes, Device *dev)
      : devicePtr(M::copyRCRef(dev)), bytes(bytes) {}

  // TODO: decide if this is needed for non-CPU
  virtual void *release() = 0;

  virtual ~DeviceTensor() = default;
};

struct CPUDeviceTensor final : DeviceTensor {
  M::LLCL::Runtime *runtime;

  CPUDeviceTensor(size_t bytes, Device *dev, M::LLCL::Runtime *runtime)
      : DeviceTensor(bytes, dev), runtime(runtime) {
    data = devicePtr->allocateData(bytes);
  }

  void *release() override {
    auto tmp = data;
    data = nullptr;
    return tmp;
  }

  ~CPUDeviceTensor() final {
    if (data) {
      devicePtr->freeData(data);
    }
  }
};

} // namespace Driver
} // namespace M

using namespace M::Driver;

MODULAR_API_EXPORT Device *M_createCPUDevice(size_t numa_id) {
  auto devPtr = M::RCRef<CPUDevice>::create(numa_id);
  return devPtr.release();
}

MODULAR_API_EXPORT Device *M_copyDevice(Device *devPtr) {
  auto newRef = M::copyRCRef(devPtr);
  return newRef.release();
}

MODULAR_API_EXPORT void M_freeDeviceData(Device *ptr, void *data) {
  ptr->freeData(data);
}

MODULAR_API_EXPORT void M_destroyDevice(Device *devPtr) {
  (void)M::takeRCRef(devPtr);
}

MODULAR_API_EXPORT const char *M_getDeviceDesc(const Device *devPtr) {
  return devPtr->getString().c_str();
}

MODULAR_API_EXPORT DeviceTensor *
M_createDeviceTensor(size_t bytes, Device *devPtr, M::LLCL::Runtime *runtime) {
  return new CPUDeviceTensor(bytes, devPtr, runtime);
}

MODULAR_API_EXPORT Device *
M_getDeviceFromDeviceTensor(DeviceTensor *tensorPtr) {
  return tensorPtr->devicePtr.getPointer();
}

MODULAR_API_EXPORT void *M_takeDataFromDeviceTensor(DeviceTensor *tensorPtr) {
  return tensorPtr->release();
}

MODULAR_API_EXPORT void M_destroyDeviceTensor(DeviceTensor *tensorPtr) {
  delete tensorPtr;
}
