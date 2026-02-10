//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Tests for debug allocation poison fill functionality.
//
//===----------------------------------------------------------------------===//

#include "GenericML/DeviceDriver/Driver.h"
#include "MLRT/unittests/Driver/TestUtils.h"
#include "Support/ML/DType.h"
#include "Support/ML/TensorSpec.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>

using namespace M;
using namespace M::GenericML::Driver;

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
    auto deviceOrErr = M::GenericML::Driver::makeCPUDevice();
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

    // Expect qNaN pattern for device allocation
    const llvm::fltSemantics *sem = spec.getEltType().getFloatSemantics();
    llvm::APFloat nan = llvm::APFloat::getQNaN(*sem);
    uint16_t expectedPattern =
        static_cast<uint16_t>(nan.bitcastToAPInt().getZExtValue());

    check16BitPattern(hostMem->data(), numElements, expectedPattern);
  } else {
    // No GPU: test host allocation with 0xFF pattern on CPU device
    auto cpu = makeCPU();

    ASSIGN_NO_ERR(DeviceMemory *, hostMem,
                  cpu->createDeviceMemory(spec, /*hostAlloc=*/true));
    std::unique_ptr<DeviceMemory> hostMemGuard(hostMem);

    check16BitPattern(hostMem->data(), numElements, 0xFFFF);
  }
}

} // namespace
