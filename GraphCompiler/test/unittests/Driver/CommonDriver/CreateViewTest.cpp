//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Tests for DeviceMemory::createView byte offset handling.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Driver/Device/Driver.h"
#include "MLRT/unittests/Driver/TestUtils.h"
#include "Support/ML/DType.h"
#include "Support/ML/TensorSpec.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <memory>

using namespace M;
using namespace M::GraphCompiler::Driver;

namespace {

class CreateViewTest : public ::testing::Test {
protected:
  static RCRef<Device> makeCPU() {
    auto deviceOrErr = M::GraphCompiler::Driver::makeCPUDevice();
    assert(!deviceOrErr.isError());
    return RCRef<Device>::copy((*deviceOrErr).get());
  }
};

// Verify that createView with a byte offset creates a valid sub-buffer and
// that the bounds check uses byte semantics correctly.
TEST_F(CreateViewTest, ByteOffsetBasic) {
  auto cpu = makeCPU();

  TensorSpec fullSpec({16}, DType::f32);
  ASSIGN_NO_ERR(DeviceMemory *, mem,
                cpu->createDeviceMemory(fullSpec, /*hostAlloc=*/true));
  std::unique_ptr<DeviceMemory> memGuard(mem);

  // Fill with a known pattern: element[i] = float(i)
  auto *data = static_cast<float *>(mem->data());
  for (int i = 0; i < 16; ++i)
    data[i] = static_cast<float>(i);

  // Create a view of the last 8 elements (byte offset = 8 * 4 = 32)
  TensorSpec viewSpec({8}, DType::f32);
  size_t byteOffset = 8 * sizeof(float);
  ASSIGN_NO_ERR(DeviceMemory *, view, mem->createView(viewSpec, byteOffset));
  std::unique_ptr<DeviceMemory> viewGuard(view);

  auto *viewData = static_cast<float *>(view->data());
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(viewData[i], static_cast<float>(i + 8))
        << "View element " << i << " should map to original element " << i + 8;
  }
}

// Verify that creating a view that would exceed the buffer returns an error.
TEST_F(CreateViewTest, ByteOffsetOutOfBounds) {
  auto cpu = makeCPU();

  TensorSpec fullSpec({16}, DType::f32);
  ASSIGN_NO_ERR(DeviceMemory *, mem,
                cpu->createDeviceMemory(fullSpec, /*hostAlloc=*/true));
  std::unique_ptr<DeviceMemory> memGuard(mem);

  // Request a view of 8 floats at an offset that would go past the end.
  // 13 * 4 = 52 byte offset, 8 * 4 = 32 byte view -> 84 > 64 total bytes
  TensorSpec viewSpec({8}, DType::f32);
  size_t byteOffset = 13 * sizeof(float);
  auto viewOrErr = mem->createView(viewSpec, byteOffset);
  EXPECT_TRUE(viewOrErr.isError());
}

// The core bug scenario: a large buffer allocated with one dtype, viewed as a
// different dtype (e.g. bf16 -> uint8), then sliced. The byte offset must be
// computed from the view's dtype, not the storage's dtype.
TEST_F(CreateViewTest, CrossDtypeViewOffset) {
  auto cpu = makeCPU();

  // Allocate a buffer of 8 bf16 elements = 16 bytes
  TensorSpec bf16Spec({8}, DType::bf16);
  ASSIGN_NO_ERR(DeviceMemory *, mem,
                cpu->createDeviceMemory(bf16Spec, /*hostAlloc=*/true));
  std::unique_ptr<DeviceMemory> memGuard(mem);

  // Fill with a known byte pattern
  std::memset(mem->data(), 0, 16);
  auto *bytes = static_cast<uint8_t *>(mem->data());
  for (int i = 0; i < 16; ++i)
    bytes[i] = static_cast<uint8_t>(i);

  // Reinterpret as 16 uint8 elements - same underlying storage
  // Now create a view of the last 4 bytes at byte offset 12
  TensorSpec u8ViewSpec({4}, DType::ui8);
  size_t byteOffset = 12;
  ASSIGN_NO_ERR(DeviceMemory *, view, mem->createView(u8ViewSpec, byteOffset));
  std::unique_ptr<DeviceMemory> viewGuard(view);

  auto *viewData = static_cast<uint8_t *>(view->data());
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(viewData[i], static_cast<uint8_t>(i + 12))
        << "Cross-dtype view element " << i;
  }
}

// Verify that a zero byte offset with matching spec returns the same buffer.
TEST_F(CreateViewTest, ZeroOffsetSameSpec) {
  auto cpu = makeCPU();

  TensorSpec spec({4}, DType::f32);
  ASSIGN_NO_ERR(DeviceMemory *, mem,
                cpu->createDeviceMemory(spec, /*hostAlloc=*/true));
  std::unique_ptr<DeviceMemory> memGuard(mem);

  ASSIGN_NO_ERR(DeviceMemory *, view, mem->createView(spec, 0));
  std::unique_ptr<DeviceMemory> viewGuard(view);

  EXPECT_EQ(view->data(), mem->data());
  EXPECT_EQ(view->bytecount(), mem->bytecount());
}

} // namespace
