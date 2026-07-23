//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Tests for debug allocation poison fill functionality.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Driver/Device/Driver.h"
#include "MLRT/unittests/Driver/TestUtils.h"
#include "Support/ML/DType.h"
#include "Support/ML/TensorSpec.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>

using namespace M;
using namespace M::GraphCompiler::Driver;

namespace {

class DebugAllocPoisonTest : public ::testing::Test {
protected:
  static ErrorOr<RCRef<Device>> makeGPU() {
    Status status;
    Device *dev = M_createAcceleratorDevice(0, &status);
    if (status.errorOr.isError())
      return status.errorOr.takeError();
    return M::takeRCRef(dev);
  }

  static RCRef<Device> makeCPU() {
    auto deviceOrErr = M::GraphCompiler::Driver::makeCPUDevice();
    assert(!deviceOrErr.isError());
    return RCRef<Device>::copy((*deviceOrErr).get());
  }

  static void check16BitPattern(const void *data, size_t count,
                                uint16_t pattern) {
    const uint16_t *ptr = static_cast<const uint16_t *>(data);
    for (size_t i = 0; i < count; ++i) {
      EXPECT_EQ(ptr[i], pattern) << "Element " << i << " doesn't match pattern";
    }
  }
};

TEST_F(DebugAllocPoisonTest, BFloat16) {
  auto gpuOrErr = makeGPU();

  constexpr size_t numElements = 16;
  TensorSpec spec({numElements}, DType::bf16);

  // Both host and device fills use the largest-finite bit pattern (e.g.
  // 0x7F7F for bf16); a non-NaN, non-Inf value chosen so the poison fill
  // doesn't conflict with the nan-check debug pass.
  const llvm::fltSemantics *sem = spec.getEltType().getFloatSemantics();
  llvm::APFloat largest = llvm::APFloat::getLargest(*sem);
  uint16_t expectedPattern =
      static_cast<uint16_t>(largest.bitcastToAPInt().getZExtValue());

  if (!gpuOrErr.isError()) {
    auto gpu = gpuOrErr.takeValue();

    ASSIGN_NO_ERR(DeviceMemory *, gpuMem,
                  gpu->createDeviceMemory(spec, /*hostAlloc=*/false));
    std::unique_ptr<DeviceMemory> gpuMemGuard(gpuMem);

    // Create host memory for verification
    ASSIGN_NO_ERR(DeviceMemory *, hostMemPtr,
                  gpu->createDeviceMemory(spec, /*hostAlloc=*/true));
    std::unique_ptr<DeviceMemory> hostMem(hostMemPtr);

    // Copy from GPU to host and synchronize
    EXPECT_NO_ERR(gpuMem->copyTo(*hostMem));
    EXPECT_NO_ERR(gpu->synchronize());

    check16BitPattern(hostMem->data(), numElements, expectedPattern);
  } else {
    auto cpu = makeCPU();

    ASSIGN_NO_ERR(DeviceMemory *, hostMem,
                  cpu->createDeviceMemory(spec, /*hostAlloc=*/true));
    std::unique_ptr<DeviceMemory> hostMemGuard(hostMem);

    check16BitPattern(hostMem->data(), numElements, expectedPattern);
  }
}

} // namespace
