//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/lib/Driver/DeviceContext/Metal/MetalDeviceContext.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"

// Suppress warnings from metal-cpp headers
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wdtor-name"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wundef"

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"

#pragma clang diagnostic pop

#include "TestUtils.h"
#include "gtest/gtest.h"

#include <algorithm> // for std::min
#include <chrono>
#include <cmath> // for std::ceil, std::sqrt
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace M::Driver {

// Flag to control performance test output
static constexpr bool kPrintPerformanceMetrics =
    false; // Set to false to disable performance output

// Helper struct to hold timing breakdowns
struct TimingBreakdown {
  double dataTransferToDevice;
  double computation;
  double dataTransferFromDevice;
  double total;
};

// Define size ranges for different tests
struct SizeRange {
  const char *name;
  std::vector<size_t> sizes;
  const char *description;
};

static const std::vector<SizeRange> SIZE_RANGES = {
    {"Small",
     {64 * 1024, 256 * 1024, 1024 * 1024}, // 64KB to 1MB
     "Small vectors (64KB - 1MB) - Testing CPU vs GPU overhead"},
    {"Medium",
     {4 * 1024 * 1024, 16 * 1024 * 1024}, // 4MB to 16MB
     "Medium vectors (4MB - 16MB) - Testing typical workloads"},
    {"Large",
     {32 * 1024 * 1024, 64 * 1024 * 1024}, // 32MB to 64MB
     "Large vectors (32MB - 64MB) - Testing memory bandwidth"}};

// Base class for Metal device tests
class MetalDeviceContextTestBase : public ::testing::Test {
protected:
  void SetUp() override {
    ErrorOr<DeviceRef> deviceOr = Device::create(Device::metalAPI);
    if (deviceOr.isError()) {
      GTEST_SKIP() << "Failed to create Metal device: " << deviceOr.getError();
      return;
    }
    device_ = std::move(*deviceOr);

    // Create a context for the device
    ErrorOr<DeviceContextRef> ctx = device_->createContext();
    if (ctx.isError()) {
      GTEST_SKIP() << "Failed to create device context: " << ctx.getError();
      return;
    }
    ctx_ = std::move(*ctx);

    // Metal device context is ready to use
  }

  void TearDown() override {
    ctx_.reset();
    device_.reset();
  }

  // Helper function to get Metal shader source that emits a known marker via
  // os_log (MSL 3.1 / macOS 14+). Compilation is attempted at test time; if
  // os_log is unavailable the load will fail and the test skips.
  std::string getPrintKernelSource() {
    return R"(
      #include <metal_stdlib>
      using namespace metal;

      kernel void print_test_kernel(uint tid [[thread_position_in_grid]])
      {
          if (tid == 0)
              os_log("gpu_print_test_marker\n");
      }
    )";
  }

  // Helper function to get Metal shader source code
  std::string getMetalShaderSource() {
    return R"(
      #include <metal_stdlib>
      using namespace metal;

      struct FloatBufPtr      { device float*       ptr; };
      struct ConstFloatBufPtr { device const float* ptr; };

      kernel void vec_add_msl(
          device FloatBufPtr&       out_ref [[buffer(0)]],
          device ConstFloatBufPtr&  in0_ref [[buffer(1)]],
          device ConstFloatBufPtr&  in1_ref [[buffer(2)]],
          constant uint&            length  [[buffer(3)]],
          uint tid [[thread_position_in_grid]])
      {
          device float*       output = out_ref.ptr;
          device const float* input0 = in0_ref.ptr;
          device const float* input1 = in1_ref.ptr;
          if (tid < length)
              output[tid] = input0[tid] + input1[tid];
      }
    )";
  }

  DeviceRef device_;
  DeviceContextRef ctx_;
};

// Test class for MSL-specific tests
class MSLDeviceContextTest : public MetalDeviceContextTestBase {};

// Test fixture is needed to enable testing of private methods in DeviceContext.
class MetalDeviceContextTest : public ::testing::Test {
protected:
  void SetUp() override {
    // No common setup needed.
  }
};

TEST_F(MetalDeviceContextTest, Smoke) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::metalAPI));

  std::string deviceName = ctx->device().name();
  EXPECT_NE(deviceName.find("Apple"), std::string::npos) << deviceName;
  std::cerr << "Name of default MTL device: " << deviceName << "\n";

  EXPECT_EQ(ctx->device().api(), Device::metalAPI);
}

TEST_F(MetalDeviceContextTest, NumDevices) {
  int num = Device::numberOfDevices(Device::metalAPI);
  EXPECT_EQ(num, 1);
  std::cerr << "Number of GPU devices: " << num << "\n";
}

TEST_F(MetalDeviceContextTestBase, BinaryMetallibExecution) {
  ASSERT_TRUE(ctx_);

  // Get device info
  const MetalDevice &metalDevice = MetalDevice::cast(*device_);
  MTL::Device *mtlDevice = metalDevice.metalDevice();
  ASSERT_TRUE(mtlDevice) << "Failed to get Metal device";

  // Get Metal shader source code
  std::string metalSource = getMetalShaderSource();
  ASSERT_FALSE(metalSource.empty()) << "Failed to get Metal shader source";

  // Load the function from MSL source code
  ErrorOr<DeviceFunctionRef> libFunctionOr =
      ctx_->loadFunction("vec_add_module", "vec_add_msl", metalSource);
  ASSERT_FALSE(libFunctionOr.isError())
      << "Failed to load Metal function: " << libFunctionOr.getError();
  DeviceFunctionRef libFunction = libFunctionOr.takeValue();

  // Test the function with some data
  const size_t length = 1024;
  const size_t elemSize = sizeof(float);

  // Create device buffers
  ErrorOr<DeviceBufferRef> in0DeviceOr =
      ctx_->enqueueCreateBuffer(length * elemSize);
  ASSERT_FALSE(in0DeviceOr.isError()) << "Failed to create in0Device buffer";
  DeviceBufferRef in0Device = in0DeviceOr.takeValue();

  ErrorOr<DeviceBufferRef> in1DeviceOr =
      ctx_->enqueueCreateBuffer(length * elemSize);
  ASSERT_FALSE(in1DeviceOr.isError()) << "Failed to create in1Device buffer";
  DeviceBufferRef in1Device = in1DeviceOr.takeValue();

  ErrorOr<DeviceBufferRef> outDeviceOr =
      ctx_->enqueueCreateBuffer(length * elemSize);
  ASSERT_FALSE(outDeviceOr.isError()) << "Failed to create outDevice buffer";
  DeviceBufferRef outDevice = outDeviceOr.takeValue();

  // Initialize host buffers
  float *in0Host = new float[length];
  float *in1Host = new float[length];
  float *outHost = new float[length];

  for (size_t i = 0; i < length; ++i) {
    in0Host[i] = static_cast<float>(i);
    in1Host[i] = 2.0f;
    outHost[i] = 0.0f;
  }

  // Copy data to device
  EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in0Device, in0Host));
  EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in1Device, in1Host));
  EXPECT_NO_ERR(ctx_->synchronize());

  void *outPtr = outDevice->ptr();
  void *in0Ptr = in0Device->ptr();
  void *in1Ptr = in1Device->ptr();
  uint32_t lengthValue = static_cast<uint32_t>(length);
  void *kernelArgs[] = {&outPtr, &in0Ptr, &in1Ptr, &lengthValue};
  size_t argSizes[] = {sizeof(outPtr), sizeof(in0Ptr), sizeof(in1Ptr),
                       sizeof(lengthValue)};
  bool argIsDevicePtr[] = {true, true, true, false};
  DeviceBuffer *kernelBuffers[] = {
      outDevice.getPointer(), in0Device.getPointer(), in1Device.getPointer()};
  MetalEnqueueFunctionArgs metalArgs = {
      kernelArgs, argSizes, argIsDevicePtr, kernelBuffers,
      sizeof(kernelBuffers) / sizeof(DeviceBuffer *)};
  void *args[] = {&metalArgs};

  // Execute kernel
  constexpr uint32_t blockDim = 256;
  EXPECT_NO_ERR(ctx_->enqueueFunctionExecDirect(
      *libFunction, length / blockDim, 1, 1, blockDim, 1, 1, 0, nullptr, 0,
      args, sizeof(kernelArgs) / sizeof(void *), argSizes));

  // Copy results back
  EXPECT_NO_ERR(ctx_->enqueueCopyFromDevice(outHost, *outDevice));
  EXPECT_NO_ERR(ctx_->synchronize());

  // Verify results
  for (size_t i = 0; i < std::min(size_t(5), length); ++i) {
    float expected = static_cast<float>(i) + 2.0f;
    EXPECT_FLOAT_EQ(outHost[i], expected)
        << "Mismatch at index " << i << ": expected " << expected << ", got "
        << outHost[i];
  }

  // Cleanup
  delete[] in0Host;
  delete[] in1Host;
  delete[] outHost;
}

TEST_F(MetalDeviceContextTestBase, DriverVersion) {
  ASSERT_TRUE(device_);
  auto driverVersion = device_->getDriverVersion();
  ASSERT_FALSE(driverVersion.isError());
  std::cout << "Metal Driver Version: " << driverVersion->toString() << "\n";
  ASSERT_GT(driverVersion->major(), 0);
  ASSERT_GE(driverVersion->minor(), 0);
  ASSERT_GE(driverVersion->patch(), 0);
}

TEST_F(MetalDeviceContextTestBase, BinaryMetallibExecutionMultipleSizes) {
  ASSERT_TRUE(ctx_);

  // Get device info
  const MetalDevice &metalDevice = MetalDevice::cast(*device_);
  MTL::Device *mtlDevice = metalDevice.metalDevice();
  ASSERT_TRUE(mtlDevice) << "Failed to get Metal device";

  // Get Metal shader source code
  std::string metalSource = getMetalShaderSource();
  ASSERT_FALSE(metalSource.empty()) << "Failed to get Metal shader source";

  // Load the function from MSL source code
  ErrorOr<DeviceFunctionRef> libFunctionOr =
      ctx_->loadFunction("vec_add_module", "vec_add_msl", metalSource);
  ASSERT_FALSE(libFunctionOr.isError())
      << "Failed to load Metal function: " << libFunctionOr.getError();
  DeviceFunctionRef libFunction = libFunctionOr.takeValue();

  // Test different vector sizes
  std::vector<size_t> testSizes = {256, 1024, 4096, 16384, 65536};
  const uint32_t blockDim = 256;

  for (size_t length : testSizes) {
    if (kPrintPerformanceMetrics) {
      std::cout << "Testing vector size: " << length << std::endl;
    }

    const size_t elemSize = sizeof(float);

    // Create device buffers
    ErrorOr<DeviceBufferRef> in0DeviceOr =
        ctx_->enqueueCreateBuffer(length * elemSize);
    ASSERT_FALSE(in0DeviceOr.isError())
        << "Failed to create in0Device buffer for size " << length;
    DeviceBufferRef in0Device = in0DeviceOr.takeValue();

    ErrorOr<DeviceBufferRef> in1DeviceOr =
        ctx_->enqueueCreateBuffer(length * elemSize);
    ASSERT_FALSE(in1DeviceOr.isError())
        << "Failed to create in1Device buffer for size " << length;
    DeviceBufferRef in1Device = in1DeviceOr.takeValue();

    ErrorOr<DeviceBufferRef> outDeviceOr =
        ctx_->enqueueCreateBuffer(length * elemSize);
    ASSERT_FALSE(outDeviceOr.isError())
        << "Failed to create outDevice buffer for size " << length;
    DeviceBufferRef outDevice = outDeviceOr.takeValue();

    // Initialize host buffers
    std::vector<float> in0Host(length);
    std::vector<float> in1Host(length);
    std::vector<float> outHost(length);

    for (size_t i = 0; i < length; ++i) {
      in0Host[i] = static_cast<float>(i);
      in1Host[i] = 2.0f;
      outHost[i] = 0.0f;
    }

    // Copy data to device
    EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in0Device, in0Host.data()));
    EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in1Device, in1Host.data()));
    EXPECT_NO_ERR(ctx_->synchronize());

    void *outPtr = outDevice->ptr();
    void *in0Ptr = in0Device->ptr();
    void *in1Ptr = in1Device->ptr();
    uint32_t lengthValue = static_cast<uint32_t>(length);
    void *kernelArgs[] = {&outPtr, &in0Ptr, &in1Ptr, &lengthValue};
    size_t argSizes[] = {sizeof(outPtr), sizeof(in0Ptr), sizeof(in1Ptr),
                         sizeof(lengthValue)};
    bool argIsDevicePtr[] = {true, true, true, false};
    DeviceBuffer *kernelBuffers[] = {
        outDevice.getPointer(), in0Device.getPointer(), in1Device.getPointer()};
    MetalEnqueueFunctionArgs metalArgs = {
        kernelArgs, argSizes, argIsDevicePtr, kernelBuffers,
        sizeof(kernelBuffers) / sizeof(DeviceBuffer *)};
    void *args[] = {&metalArgs};

    // Execute kernel
    uint32_t gridSize =
        static_cast<uint32_t>((length + blockDim - 1) / blockDim);
    EXPECT_NO_ERR(ctx_->enqueueFunctionExecDirect(
        *libFunction, gridSize, 1, 1, blockDim, 1, 1, 0, nullptr, 0, args,
        sizeof(kernelArgs) / sizeof(void *), argSizes));
    EXPECT_NO_ERR(ctx_->synchronize());

    // Copy results back
    EXPECT_NO_ERR(ctx_->enqueueCopyFromDevice(outHost.data(), *outDevice));
    EXPECT_NO_ERR(ctx_->synchronize());

    // Verify results for a sample of elements
    // check all elements
    for (size_t i = 0; i < length; ++i) {
      float expected = static_cast<float>(i) + 2.0f;
      EXPECT_FLOAT_EQ(outHost[i], expected)
          << "Mismatch at index " << i << " for size " << length
          << ": expected " << expected << ", got " << outHost[i];
    }

    if (kPrintPerformanceMetrics) {
      std::cout << "✓ Vector size " << length << " passed" << std::endl;
    }
  }
}

TEST_F(MetalDeviceContextTestBase, BinaryMetallibExecutionPerformance) {
  ASSERT_TRUE(ctx_);

  // Get device info
  const MetalDevice &metalDevice = MetalDevice::cast(*device_);
  MTL::Device *mtlDevice = metalDevice.metalDevice();
  ASSERT_TRUE(mtlDevice) << "Failed to get Metal device";

  // Get Metal shader source code
  std::string metalSource = getMetalShaderSource();
  ASSERT_FALSE(metalSource.empty()) << "Failed to get Metal shader source";

  // Load the function from MSL source code
  ErrorOr<DeviceFunctionRef> libFunctionOr =
      ctx_->loadFunction("vec_add_module", "vec_add_msl", metalSource);
  ASSERT_FALSE(libFunctionOr.isError())
      << "Failed to load Metal function: " << libFunctionOr.getError();
  DeviceFunctionRef libFunction = libFunctionOr.takeValue();

  // Performance test with larger vectors
  std::vector<size_t> performanceSizes = {1024, 16384, 65536, 1048576};
  const uint32_t blockDim = 256;
  const int numRuns = 5; // Run multiple times for better average

  if (kPrintPerformanceMetrics) {
    std::cout << "\n=== Metal GPU Vector Addition Performance ===\n";
    std::cout << std::setw(12) << "Size" << " | " << std::setw(15)
              << "Avg Time (ms)"
              << " | " << std::setw(15) << "Throughput (GB/s)" << "\n";
    std::cout << std::string(50, '-') << "\n";
  }

  for (size_t length : performanceSizes) {
    const size_t elemSize = sizeof(float);
    const size_t totalBytes =
        length * elemSize * 3; // 3 buffers (2 input, 1 output)

    double totalExecutionTime = 0.0;

    for (int run = 0; run < numRuns; ++run) {
      // Create device buffers
      ErrorOr<DeviceBufferRef> in0DeviceOr =
          ctx_->enqueueCreateBuffer(length * elemSize);
      ASSERT_FALSE(in0DeviceOr.isError())
          << "Failed to create in0Device buffer";
      DeviceBufferRef in0Device = in0DeviceOr.takeValue();

      ErrorOr<DeviceBufferRef> in1DeviceOr =
          ctx_->enqueueCreateBuffer(length * elemSize);
      ASSERT_FALSE(in1DeviceOr.isError())
          << "Failed to create in1Device buffer";
      DeviceBufferRef in1Device = in1DeviceOr.takeValue();

      ErrorOr<DeviceBufferRef> outDeviceOr =
          ctx_->enqueueCreateBuffer(length * elemSize);
      ASSERT_FALSE(outDeviceOr.isError())
          << "Failed to create outDevice buffer";
      DeviceBufferRef outDevice = outDeviceOr.takeValue();

      // Initialize host buffers
      std::vector<float> in0Host(length);
      std::vector<float> in1Host(length);
      std::vector<float> outHost(length);

      for (size_t i = 0; i < length; ++i) {
        in0Host[i] = static_cast<float>(i);
        in1Host[i] = 2.0f;
        outHost[i] = 0.0f;
      }

      // Copy data to device
      EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in0Device, in0Host.data()));
      EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in1Device, in1Host.data()));
      EXPECT_NO_ERR(ctx_->synchronize());

      void *outPtr = outDevice->ptr();
      void *in0Ptr = in0Device->ptr();
      void *in1Ptr = in1Device->ptr();
      uint32_t lengthValue = static_cast<uint32_t>(length);
      void *kernelArgs[] = {&outPtr, &in0Ptr, &in1Ptr, &lengthValue};
      size_t argSizes[] = {sizeof(outPtr), sizeof(in0Ptr), sizeof(in1Ptr),
                           sizeof(lengthValue)};
      bool argIsDevicePtr[] = {true, true, true, false};
      DeviceBuffer *kernelBuffers[] = {outDevice.getPointer(),
                                       in0Device.getPointer(),
                                       in1Device.getPointer()};
      MetalEnqueueFunctionArgs metalArgs = {
          kernelArgs, argSizes, argIsDevicePtr, kernelBuffers,
          sizeof(kernelBuffers) / sizeof(DeviceBuffer *)};
      void *args[] = {&metalArgs};

      // Time the kernel execution only
      std::chrono::high_resolution_clock::time_point startTime =
          std::chrono::high_resolution_clock::now();

      uint32_t gridSize =
          static_cast<uint32_t>((length + blockDim - 1) / blockDim);
      EXPECT_NO_ERR(ctx_->enqueueFunctionExecDirect(
          *libFunction, gridSize, 1, 1, blockDim, 1, 1, 0, nullptr, 0, args,
          sizeof(kernelArgs) / sizeof(void *), argSizes));
      EXPECT_NO_ERR(ctx_->synchronize());

      std::chrono::high_resolution_clock::time_point endTime =
          std::chrono::high_resolution_clock::now();

      // Copy results back (not timed)
      EXPECT_NO_ERR(ctx_->enqueueCopyFromDevice(outHost.data(), *outDevice));
      EXPECT_NO_ERR(ctx_->synchronize());

      // Verify correctness for first run
      if (run == 0) {
        // check all elements
        for (size_t i = 0; i < length; ++i) {
          float expected = static_cast<float>(i) + 2.0f;
          EXPECT_FLOAT_EQ(outHost[i], expected)
              << "Mismatch at index " << i << " for size " << length;
        }
      }

      std::chrono::microseconds duration =
          std::chrono::duration_cast<std::chrono::microseconds>(endTime -
                                                                startTime);
      totalExecutionTime +=
          duration.count() / 1000.0; // Convert to milliseconds
    }

    double avgTime = totalExecutionTime / numRuns;
    double throughput = (totalBytes / 1e9) / (avgTime / 1000.0); // GB/s

    if (kPrintPerformanceMetrics) {
      std::cout << std::setw(12) << length << " | " << std::setw(15)
                << std::fixed << std::setprecision(3) << avgTime << " | "
                << std::setw(15) << std::setprecision(2) << throughput << "\n";
    }
  }

  if (kPrintPerformanceMetrics) {
    std::cout << std::string(50, '-') << "\n\n";
  }
}

TEST_F(MetalDeviceContextTestBase, BinaryMetallibExecutionCPUComparison) {
  ASSERT_TRUE(ctx_);

  // Get device info
  const MetalDevice &metalDevice = MetalDevice::cast(*device_);
  MTL::Device *mtlDevice = metalDevice.metalDevice();
  ASSERT_TRUE(mtlDevice) << "Failed to get Metal device";

  // Get Metal shader source code
  std::string metalSource = getMetalShaderSource();
  ASSERT_FALSE(metalSource.empty()) << "Failed to get Metal shader source";

  // Load the function from MSL source code
  ErrorOr<DeviceFunctionRef> libFunctionOr =
      ctx_->loadFunction("vec_add_module", "vec_add_msl", metalSource);
  ASSERT_FALSE(libFunctionOr.isError())
      << "Failed to load Metal function: " << libFunctionOr.getError();
  DeviceFunctionRef libFunction = libFunctionOr.takeValue();

  // Comparison test with different vector sizes
  std::vector<size_t> comparisonSizes = {1024, 16384, 65536, 1048576};
  const uint32_t blockDim = 256;
  const int numRuns = 10; // More runs for better averaging

  if (kPrintPerformanceMetrics) {
    std::cout << "\n=== CPU vs Metal GPU Vector Addition Performance "
                 "Comparison ===\n";
    std::cout << std::setw(12) << "Size" << " | " << std::setw(12) << "CPU (ms)"
              << " | " << std::setw(12) << "GPU (ms)" << " | " << std::setw(12)
              << "Speedup"
              << " | " << std::setw(15) << "CPU (GB/s)" << " | "
              << std::setw(15) << "GPU (GB/s)" << "\n";
    std::cout << std::string(90, '-') << "\n";
  }

  for (size_t length : comparisonSizes) {
    const size_t elemSize = sizeof(float);
    const size_t totalBytes =
        length * elemSize * 3; // 3 buffers (2 input, 1 output)

    // Prepare host data (shared between CPU and GPU tests)
    std::vector<float> in0Host(length);
    std::vector<float> in1Host(length);
    std::vector<float> cpuResult(length);
    std::vector<float> gpuResult(length);

    for (size_t i = 0; i < length; ++i) {
      in0Host[i] = static_cast<float>(i);
      in1Host[i] = 2.0f;
      cpuResult[i] = 0.0f;
      gpuResult[i] = 0.0f;
    }

    // === CPU BENCHMARK ===
    double totalCpuTime = 0.0;
    for (int run = 0; run < numRuns; ++run) {
      std::chrono::high_resolution_clock::time_point startTime =
          std::chrono::high_resolution_clock::now();

      // Simple CPU vector addition
      for (size_t i = 0; i < length; ++i) {
        cpuResult[i] = in0Host[i] + in1Host[i];
      }

      std::chrono::high_resolution_clock::time_point endTime =
          std::chrono::high_resolution_clock::now();
      std::chrono::microseconds duration =
          std::chrono::duration_cast<std::chrono::microseconds>(endTime -
                                                                startTime);
      totalCpuTime += duration.count() / 1000.0; // Convert to milliseconds
    }
    double avgCpuTime = totalCpuTime / numRuns;

    // === GPU BENCHMARK ===
    double totalGpuTime = 0.0;
    for (int run = 0; run < numRuns; ++run) {
      // Create device buffers
      ErrorOr<DeviceBufferRef> in0DeviceOr =
          ctx_->enqueueCreateBuffer(length * elemSize);
      ASSERT_FALSE(in0DeviceOr.isError())
          << "Failed to create in0Device buffer";
      DeviceBufferRef in0Device = in0DeviceOr.takeValue();

      ErrorOr<DeviceBufferRef> in1DeviceOr =
          ctx_->enqueueCreateBuffer(length * elemSize);
      ASSERT_FALSE(in1DeviceOr.isError())
          << "Failed to create in1Device buffer";
      DeviceBufferRef in1Device = in1DeviceOr.takeValue();

      ErrorOr<DeviceBufferRef> outDeviceOr =
          ctx_->enqueueCreateBuffer(length * elemSize);
      ASSERT_FALSE(outDeviceOr.isError())
          << "Failed to create outDevice buffer";
      DeviceBufferRef outDevice = outDeviceOr.takeValue();

      // Copy data to device
      EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in0Device, in0Host.data()));
      EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in1Device, in1Host.data()));
      EXPECT_NO_ERR(ctx_->synchronize());

      // Mimic how Mojo passes pointers
      void *outPtr = outDevice->ptr();
      void *in0Ptr = in0Device->ptr();
      void *in1Ptr = in1Device->ptr();
      uint32_t lengthValue = static_cast<uint32_t>(length);
      void *kernelArgs[] = {&outPtr, &in0Ptr, &in1Ptr, &lengthValue};
      size_t argSizes[] = {sizeof(outPtr), sizeof(in0Ptr), sizeof(in1Ptr),
                           sizeof(lengthValue)};
      bool argIsDevicePtr[] = {true, true, true, false};
      DeviceBuffer *kernelBuffers[] = {outDevice.getPointer(),
                                       in0Device.getPointer(),
                                       in1Device.getPointer()};
      MetalEnqueueFunctionArgs metalArgs = {
          kernelArgs, argSizes, argIsDevicePtr, kernelBuffers,
          sizeof(kernelBuffers) / sizeof(DeviceBuffer *)};
      void *args[] = {&metalArgs};

      // Time the kernel execution only
      std::chrono::high_resolution_clock::time_point startTime =
          std::chrono::high_resolution_clock::now();

      uint32_t gridSize =
          static_cast<uint32_t>((length + blockDim - 1) / blockDim);
      EXPECT_NO_ERR(ctx_->enqueueFunctionExecDirect(
          *libFunction, gridSize, 1, 1, blockDim, 1, 1, 0, nullptr, 0, args,
          sizeof(kernelArgs) / sizeof(void *), argSizes));
      EXPECT_NO_ERR(ctx_->synchronize());

      std::chrono::high_resolution_clock::time_point endTime =
          std::chrono::high_resolution_clock::now();

      // Copy results back (for the last run only, to verify correctness)
      if (run == numRuns - 1) {
        EXPECT_NO_ERR(
            ctx_->enqueueCopyFromDevice(gpuResult.data(), *outDevice));
        EXPECT_NO_ERR(ctx_->synchronize());
      }

      std::chrono::microseconds duration =
          std::chrono::duration_cast<std::chrono::microseconds>(endTime -
                                                                startTime);
      totalGpuTime += duration.count() / 1000.0; // Convert to milliseconds
    }
    double avgGpuTime = totalGpuTime / numRuns;

    // === CORRECTNESS VERIFICATION ===
    // Verify both CPU and GPU produce the same results
    // check all elements
    for (size_t i = 0; i < length; ++i) {
      float expected = static_cast<float>(i) + 2.0f;
      EXPECT_FLOAT_EQ(cpuResult[i], expected)
          << "CPU mismatch at index " << i << " for size " << length;
      EXPECT_FLOAT_EQ(gpuResult[i], expected)
          << "GPU mismatch at index " << i << " for size " << length;
      EXPECT_FLOAT_EQ(cpuResult[i], gpuResult[i])
          << "CPU/GPU mismatch at index " << i << " for size " << length;
    }

    // === PERFORMANCE CALCULATIONS ===
    double speedup = avgCpuTime / avgGpuTime;
    double cpuThroughput = (totalBytes / 1e9) / (avgCpuTime / 1000.0); // GB/s
    double gpuThroughput = (totalBytes / 1e9) / (avgGpuTime / 1000.0); // GB/s

    if (kPrintPerformanceMetrics) {
      std::cout << std::setw(12) << length << " | " << std::setw(12)
                << std::fixed << std::setprecision(3) << avgCpuTime << " | "
                << std::setw(12) << std::fixed << std::setprecision(3)
                << avgGpuTime << " | " << std::setw(12) << std::fixed
                << std::setprecision(2) << speedup << "x"
                << " | " << std::setw(15) << std::fixed << std::setprecision(2)
                << cpuThroughput << " | " << std::setw(15) << std::fixed
                << std::setprecision(2) << gpuThroughput << "\n";
    }
  }

  if (kPrintPerformanceMetrics) {
    std::cout << std::string(90, '-') << "\n";
    std::cout << "Note: GPU timing includes only kernel execution, not memory "
                 "transfers.\n";
    std::cout << "      CPU uses single-threaded implementation for fair "
                 "comparison.\n\n";
  }
}
// Returns true if MTL::LogState can be created on the current platform
// (requires macOS 14+). Used to skip print tests on older systems.
static bool isLogStateSupported(MTL::Device *mtlDevice) {
  NS::SharedPtr<MTL::LogStateDescriptor> desc =
      NS::TransferPtr(MTL::LogStateDescriptor::alloc()->init());
  NS::Error *error = nullptr;
  NS::SharedPtr<MTL::LogState> ls =
      NS::TransferPtr(mtlDevice->newLogState(desc.get(), &error));
  return !error && ls;
}

// Test: GPU print output is captured on stdout when logging is enabled (the
// default). The kernel calls os_log() with a known marker string; after
// synchronisation the string must appear in stdout because the log handler
// installed by getOrCreateLogState() writes there.
TEST_F(MetalDeviceContextTestBase, MetalPrint_CapturesOutput) {
  const MetalDevice &metalDevice = MetalDevice::cast(*device_);
  if (!isLogStateSupported(metalDevice.metalDevice()))
    GTEST_SKIP() << "MTL::LogState not supported on this platform";

  ErrorOr<DeviceFunctionRef> fnOr = ctx_->loadFunction(
      "print_test_module", "print_test_kernel", getPrintKernelSource());
  if (fnOr.isError())
    GTEST_SKIP() << "os_log not available in MSL on this platform: "
                 << fnOr.getError();
  DeviceFunctionRef fn = fnOr.takeValue();

  MetalEnqueueFunctionArgs metalArgs = {nullptr, nullptr, nullptr, nullptr, 0};
  void *args[] = {&metalArgs};
  testing::internal::CaptureStdout();
  EXPECT_NO_ERR(ctx_->enqueueFunctionExecDirect(
      *fn, 1, 1, 1, 1, 1, 1, /*sharedMemBytes=*/0,
      /*attrs=*/nullptr, /*numAttrs=*/0, args));
  EXPECT_NO_ERR(ctx_->synchronize());
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("gpu_print_test_marker"), std::string::npos)
      << "Expected GPU print marker in stdout; got: " << output;
}

// Test: METAL_DISABLE_LOGGING attribute prevents the log state from being
// attached to the command buffer, so no GPU print output reaches stdout.
TEST_F(MetalDeviceContextTestBase, MetalPrint_DisabledByAttribute) {
  const MetalDevice &metalDevice = MetalDevice::cast(*device_);
  if (!isLogStateSupported(metalDevice.metalDevice()))
    GTEST_SKIP() << "MTL::LogState not supported on this platform";

  ErrorOr<DeviceFunctionRef> fnOr =
      ctx_->loadFunction("print_test_module_disabled", "print_test_kernel",
                         getPrintKernelSource());
  if (fnOr.isError())
    GTEST_SKIP() << "os_log not available in MSL on this platform: "
                 << fnOr.getError();
  DeviceFunctionRef fn = fnOr.takeValue();

  Device::Attribute attrs[] = {Device::Attribute::METAL_DISABLE_LOGGING};
  MetalEnqueueFunctionArgs metalArgs = {nullptr, nullptr, nullptr, nullptr, 0};
  void *args[] = {&metalArgs};
  testing::internal::CaptureStdout();
  EXPECT_NO_ERR(ctx_->enqueueFunctionExecDirect(
      *fn, 1, 1, 1, 1, 1, 1, /*sharedMemBytes=*/0, static_cast<void *>(attrs),
      /*numAttrs=*/1, args));
  EXPECT_NO_ERR(ctx_->synchronize());
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(output.find("gpu_print_test_marker"), std::string::npos)
      << "Expected no GPU print output with logging disabled; got: " << output;
}

// Regression test: memAlloc must donate newBuffer's +1 directly to the
// tracker. A previous version added a spurious buf->retain(), leaving the
// MTL::Buffer at refcount 2; memFree's release() then leaked it.
TEST_F(MetalDeviceContextTestBase, MemAllocDonatesRefcountToTracker) {
  ASSERT_TRUE(ctx_);
  const MetalDevice &metalDev = MetalDevice::cast(*device_);
  const size_t baseline = metalDev.getTrackedBuffers().size();

  void *ptr = nullptr;
  EXPECT_NO_ERR(
      ctx_->memAlloc(&ptr, /*byteSize=*/4096, Device::AllocMode::HOST));
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(metalDev.getTrackedBuffers().size(), baseline + 1);

  MTL::Buffer *buf = metalDev.getTrackedBuffer(ptr);
  ASSERT_NE(buf, nullptr);
  EXPECT_EQ(buf->retainCount(), 1u);

  EXPECT_NO_ERR(ctx_->memFree(ptr, /*isHost=*/true));
  EXPECT_EQ(metalDev.getTrackedBuffers().size(), baseline);
}

// Test: chunkAlloc rejects a request exactly equal to maxBufferLength.
// allocateNewBuffer uses `byteSize >= maxBufferLength`, so the boundary value
// must fail even though it is only one byte over the usable limit.
TEST_F(MetalDeviceContextTestBase, ChunkAlloc_AtMaxBufferLength_ReturnsError) {
  MetalDevice &metalDev = MetalDevice::cast(*device_);
  size_t maxLen = metalDev.metalDevice()->maxBufferLength();

  void *ptr = nullptr;
  EXPECT_ERR_MSG(metalDev.chunkAlloc(&ptr, maxLen, /*isHost=*/false),
                 "larger than allowed");
  EXPECT_EQ(ptr, nullptr);
}

// Test: chunkAlloc rejects SIZE_MAX — the largest possible size_t value.
TEST_F(MetalDeviceContextTestBase, ChunkAlloc_SizeMax_ReturnsError) {
  MetalDevice &metalDev = MetalDevice::cast(*device_);

  void *ptr = nullptr;
  EXPECT_ERR_MSG(metalDev.chunkAlloc(&ptr, SIZE_MAX, /*isHost=*/false),
                 "larger than allowed");
  EXPECT_EQ(ptr, nullptr);
}

// Test: chunkAlloc with a normal size succeeds, buffer is tracked, chunkFree
// removes it.
TEST_F(MetalDeviceContextTestBase, ChunkAlloc_ValidSize_SucceedsAndFrees) {
  MetalDevice &metalDev = MetalDevice::cast(*device_);
  const size_t baseline = metalDev.getTrackedBuffers().size();

  void *ptr = nullptr;
  EXPECT_NO_ERR(metalDev.chunkAlloc(&ptr, /*byteSize=*/4096, /*isHost=*/false));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(metalDev.getTrackedBuffers().size(), baseline + 1);
  EXPECT_NE(metalDev.getTrackedBuffer(ptr), nullptr);

  EXPECT_NO_ERR(metalDev.chunkFree(ptr, /*isHost=*/false));
  EXPECT_EQ(metalDev.getTrackedBuffers().size(), baseline);
}

// getMemoryInfo reports recommendedMaxWorkingSetSize as total and
// (total - currentAllocatedSize) as free.
TEST_F(MetalDeviceContextTestBase, GetMemInfo_TotalAndFree) {
  const MetalDevice &metalDev = MetalDevice::cast(*device_);
  MTL::Device *mtlDev = metalDev.metalDevice();
  size_t expectedTotal = mtlDev->recommendedMaxWorkingSetSize();
  size_t used = mtlDev->currentAllocatedSize();
  size_t expectedFree = expectedTotal > used ? expectedTotal - used : 0;

  ASSIGN_NO_ERR(Device::MemInfo, info, metalDev.getMemoryInfo());
  EXPECT_EQ(info.total, expectedTotal);
  EXPECT_EQ(info.free, expectedFree);
}

// maxSingleAllocationSize on Metal is maxBufferLength.
TEST_F(MetalDeviceContextTestBase,
       MaxSingleAllocationSize_MatchesBufferLength) {
  const MetalDevice &metalDev = MetalDevice::cast(*device_);
  ASSIGN_NO_ERR(size_t, actual, metalDev.maxSingleAllocationSize());
  EXPECT_EQ(actual, metalDev.metalDevice()->maxBufferLength());
  EXPECT_GT(actual, 0u);
}

} // namespace M::Driver
