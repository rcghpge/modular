//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MAXDriver/Driver.h"

struct CPUDevice {
  size_t numa_id;
};

MODULAR_API_EXPORT CPUDevice *M_createCPUDevice(size_t numa_id) {
  auto dev = new CPUDevice{numa_id};
  return dev;
}

MODULAR_API_EXPORT size_t M_getInfo(const CPUDevice *dev) {
  return dev->numa_id;
}

MODULAR_API_EXPORT void M_destroyCPUDevice(CPUDevice *dev) { delete dev; }
