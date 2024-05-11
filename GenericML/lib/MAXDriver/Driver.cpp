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
  size_t numa_id;
  std::string desc;

  Device(size_t numa_id) : numa_id(numa_id), desc(toString()) {}

  const std::string &getString() const { return desc; }

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
  M::LLCL::Runtime *runtime;

  DeviceTensor(size_t bytes, Device *dev, M::LLCL::Runtime *runtime)
      : devicePtr(M::copyRCRef(dev)), bytes(bytes), runtime(runtime) {
    data = runtime->getAllocator()->allocateBytes(bytes,
                                                  M::kPreferredMemoryAlignment);
  }

  void *release() {
    auto tmp = data;
    data = nullptr;
    return tmp;
  }

  ~DeviceTensor() {
    if (data) {
      runtime->getAllocator()->deallocateBytes(data, bytes);
    }
  }
};

} // namespace Driver
} // namespace M

using namespace M::Driver;

MODULAR_API_EXPORT Device *M_createCPUDevice(size_t numa_id) {
  auto devPtr = M::RCRef<Device>::create(numa_id);
  return devPtr.release();
}

MODULAR_API_EXPORT Device *M_copyDevice(Device *devPtr) {
  auto newRef = M::copyRCRef(devPtr);
  return newRef.release();
}

MODULAR_API_EXPORT void M_destroyDevice(Device *devPtr) {
  (void)M::takeRCRef(devPtr);
}

MODULAR_API_EXPORT const char *M_getDeviceDesc(const Device *devPtr) {
  return devPtr->getString().c_str();
}

MODULAR_API_EXPORT DeviceTensor *
M_createDeviceTensor(size_t bytes, Device *devPtr, M::LLCL::Runtime *runtime) {
  return new DeviceTensor(bytes, devPtr, runtime);
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
