//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CPU_EVENT_H
#define TARGET_LIB_CPU_EVENT_H

#include "HAL/M_driver_common.h"
#include "MLRT/AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "MLRT/AsyncRT/Runtime/CPUDevice.h"

namespace MLRT = M::MLRT;

/// A CPU event is a handle to a point in a queue's async chain. Recording the
/// event captures the queue's current chain value; synchronizing/querying it
/// awaits or inspects that async value. This mirrors M::Driver::CpuDeviceEvent.
struct M_driver_event {
  M_driver_event(MLRT::CPUDeviceRef cpuDevice, MLRT::AnyAsyncValueRef value)
      : cpuDevice(std::move(cpuDevice)), value(std::move(value)) {}

  MLRT::CPUDeviceRef cpuDevice;
  MLRT::AnyAsyncValueRef value;
};

#endif // TARGET_LIB_CPU_EVENT_H
