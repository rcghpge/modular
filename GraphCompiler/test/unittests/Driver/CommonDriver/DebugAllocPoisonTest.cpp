//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Tests for debug allocation poison fill functionality.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Driver/Device/DeviceExecution.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/unittests/Driver/TestUtils.h"
#include "Support/ML/DType.h"
#include "Support/ML/TensorSpec.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>

using namespace M;
using namespace M::GraphCompiler::Driver;

namespace {

class DebugAllocPoisonTest : public ::testing::Test {
protected:
  static ErrorOr<M::Driver::DeviceContextRef> makeGPU() {
    return M::Driver::DeviceContext::getOrCreate(M::Driver::Device::gpuAPI, 0);
  }

  static M::Driver::DeviceContextRef makeCPU() {
    auto ctxOrErr = M::Driver::DeviceContext::create(M::Driver::Device::cpuAPI);
    assert(!ctxOrErr.isError());
    return ctxOrErr.takeValue();
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

    ASSIGN_NO_ERR(M::Driver::DeviceBufferRef, gpuBuf,
                  allocateDeviceBuffer(gpu, spec, /*hostAlloc=*/false));

    // Create host memory for verification
    ASSIGN_NO_ERR(M::Driver::DeviceBufferRef, hostBuf,
                  allocateDeviceBuffer(gpu, spec, /*hostAlloc=*/true));

    // Copy from GPU to host and synchronize
    EXPECT_NO_ERR(gpuBuf->copyTo(*hostBuf));
    EXPECT_NO_ERR(gpu->synchronize());

    check16BitPattern(hostBuf->ptr(), numElements, expectedPattern);
  } else {
    auto cpu = makeCPU();

    ASSIGN_NO_ERR(M::Driver::DeviceBufferRef, hostBuf,
                  allocateDeviceBuffer(cpu, spec, /*hostAlloc=*/true));

    check16BitPattern(hostBuf->ptr(), numElements, expectedPattern);
  }
}

} // namespace
