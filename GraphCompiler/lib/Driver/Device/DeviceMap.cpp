//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Driver/Device/DeviceMap.h"
#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/AssertStream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

using namespace M::GraphCompiler::Driver;
using namespace M;

ErrorOr<llvm::SmallVector<AsyncRT::AnyAsyncValueRef>>
M::GraphCompiler::Driver::mapDeviceRefsToDevices(
    const ArrayRef<DeviceRef> deviceRefs, const DeviceMap &deviceMap,
    CPUDevice &cpuDevice) {

  // We need to explicitly synchronize the device here until MGP shares the same
  // DeviceContext as the driver. Otherwise there may be an async alloc/copy
  // pending on the driver when the first kernel tries to access the resource.
  for (auto &dev : deviceMap) {
    if (dev.second) {
      if (auto err = dev.second->synchronize())
        return err.takeError();
    }
  }

  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> deviceContexts;
  for (const auto &devRef : deviceRefs) {
    // CPU device not explicitly set so deviceMap["cpu"] may be null
    if (devRef.label == kCPULabel) {
      ErrorOr<M::Driver::DeviceContextRef> ctxOr =
          M::Driver::DeviceContext::create("cpu", 0);
      if (ctxOr.isError())
        return ctxOr.takeError();
      deviceContexts.push_back(
          AnyAsyncValueRef::createReady<M::Driver::DeviceContextRef>(
              cpuDevice, ctxOr.takeValue()));
    } else {
      if (auto it = deviceMap.find(devRef);
          it != deviceMap.end() && it->second) {
        deviceContexts.push_back(
            AnyAsyncValueRef::createReady<M::Driver::DeviceContextRef>(
                cpuDevice, it->second));
      } else {
        ASSERT_STREAM(devRef.label == kGPULabel || devRef.label == kNPULabel,
                      "Only accelerator device context supported");
        // In this case we use the device ref id as the device id. THIS IS NOT
        // THE CASE WHEN THE DEVICE IS PASSED IN TO THE GRAPH.
        // We should eventually push the device selection up the stack into the
        // mt command and remove all of the code in this else branch.
        size_t numPhysicalDevices =
            M::Driver::Device::numberOfDevices(M::Driver::Device::gpuAPI);
        if (devRef.id >= numPhysicalDevices)
          return Error(devRef.toString() +
                       " id exceeds number of available accelerator devices.");
        ErrorOr<M::Driver::DeviceContextRef> ctxOr =
            M::Driver::DeviceContext::create(M::Driver::Device::gpuAPI,
                                             devRef.id);
        if (ctxOr.isError())
          return ctxOr.takeError();
        deviceContexts.push_back(
            AnyAsyncValueRef::createReady<M::Driver::DeviceContextRef>(
                cpuDevice, ctxOr.takeValue()));
      }
    }
  }
  return deviceContexts;
}
