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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <sstream>
#include <utility>

namespace M {

namespace Driver {

struct Device;
struct DeviceMemory {
  enum DeviceMemoryKind {
    K_CPU,
  };
  DeviceMemory(DeviceMemoryKind kind, size_t bytes, Device *dev)
      : devicePtr(M::copyRCRef(dev)), bytes(bytes), kind(kind) {}

  // TODO: decide if this is needed for non-CPU
  virtual void *release() = 0;
  void *data() { return ptr; }
  size_t bytecount() const { return bytes; }
  virtual ~DeviceMemory() = default;
  DeviceMemoryKind getKind() const { return kind; }
  M::RCRef<Device> devicePtr;
  virtual ErrorOrSuccess copy_to(DeviceMemory &dst) const = 0;

protected:
  void *ptr;
  size_t bytes;
  const DeviceMemoryKind kind;
};
struct Device : M::ReferenceCounted<Device> {
  virtual const std::string &getString() const = 0;
  virtual void *allocateData(size_t bytes) = 0;
  virtual void freeData(void *ptr) = 0;
  virtual DeviceMemory *createDeviceMemory(size_t bytes) = 0;
  virtual DeviceMemory *createDeviceMemory(void *data, size_t bytes) = 0;
  virtual ~Device() = default;
};

struct CPUDeviceMemory final : DeviceMemory {

  CPUDeviceMemory(size_t bytes, Device *dev) : DeviceMemory(K_CPU, bytes, dev) {
    ptr = devicePtr->allocateData(bytes);
  }

  CPUDeviceMemory(void *ownedData, size_t bytes, Device *dev)
      : DeviceMemory(K_CPU, bytes, dev) {
    ptr = ownedData;
  }

  void *release() override {
    auto tmp = ptr;
    ptr = nullptr;
    return tmp;
  }

  ~CPUDeviceMemory() final {
    if (ptr) {
      devicePtr->freeData(ptr);
    }
  }

  static bool classof(const DeviceMemory *tensor) {
    return tensor->getKind() == K_CPU;
  }

  ErrorOrSuccess copy_to(DeviceMemory &dst) const override {
    if (bytecount() != dst.bytecount()) {
      return M::Error("src and dst bytecounts do not match");
    }
    return llvm::TypeSwitch<DeviceMemory *, M::ErrorOrSuccess>(&dst)
        .Case([&](CPUDeviceMemory *dst) {
          std::memcpy(dst->data(), ptr, bytecount());
          return M::success();
        })
        .Default([&](DeviceMemory *dst) {
          return M::Error("unknown dst DeviceMemory type");
        });
  }
};
struct CPUDevice : Device {
  size_t numa_id;
  std::string desc;

  CPUDevice(size_t numa_id) : numa_id(numa_id), desc(toString()) {}

  ~CPUDevice() override = default;

  void *allocateData(size_t bytes) override {
    return M::alignedAlloc(kPreferredMemoryAlignment, bytes);
  }

  DeviceMemory *createDeviceMemory(size_t bytes) override {
    return new CPUDeviceMemory(bytes, this);
  }

  DeviceMemory *createDeviceMemory(void *data, size_t bytes) override {
    return new CPUDeviceMemory(data, bytes, this);
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

MODULAR_API_EXPORT DeviceMemory *M_createDeviceMemory(size_t bytes,
                                                      Device *devPtr) {
  return devPtr->createDeviceMemory(bytes);
}

MODULAR_API_EXPORT DeviceMemory *
M_createDeviceMemoryFrom(void *data, size_t bytes, Device *devPtr) {
  return devPtr->createDeviceMemory(data, bytes);
}

MODULAR_API_EXPORT Device *
M_getDeviceFromDeviceMemory(DeviceMemory *tensorPtr) {
  return tensorPtr->devicePtr.getPointer();
}

MODULAR_API_EXPORT void *M_takeDataFromDeviceMemory(DeviceMemory *tensorPtr) {
  return tensorPtr->release();
}

MODULAR_API_EXPORT void M_destroyDeviceMemory(DeviceMemory *tensorPtr) {
  delete tensorPtr;
}

MODULAR_API_EXPORT void M_copyDeviceMemory(DeviceMemory *dst,
                                           const DeviceMemory *src) {
  auto errOr = src->copy_to(*dst);
  assert(!errOr.isError());
}
