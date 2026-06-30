//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "../lib/Metal/Common.h"
#include "Fixture.h"
#include "HAL/M_driver_bundle.h"
#include "HAL/M_driver_common.h"
#include <string>

#include "llvm/ADT/ScopeExit.h"

// Extend test-cases per-plugin beyond the common ones in PluginTest.cpp

/// NOTE: assumes `s` is null terminated
M_driver_slice M_driver_slice_from(const char *s) {
  uint64_t size = strlen(s);
  return M_driver_slice{s, size};
}

M_driver_slice M_driver_slice_from(std::string &s) {
  return M_driver_slice{s.data(), s.size()};
}

static std::string vecAddMsl = R"MSL(
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
)MSL";

TEST_F(DriverPluginTest, CustomStatusMessage) {
  char errorMessage[1024];
  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(driver, M_DRIVER_STATUS_UNIMPLEMENTED,
                                 errorMessage, sizeof(errorMessage)));
  ASSERT_STREQ(errorMessage, "Unimplemented");
  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(driver, M_DRIVER_STATUS_UNINITIALIZED,
                                 errorMessage, sizeof(errorMessage)));
  ASSERT_STREQ(errorMessage, "Uninitialized");
}

TEST_F(DriverPluginContextTest, FunctionLoad) {
  for (M_driver_context_t context : contexts) {
    M_driver_runtime_bundle_t library;
    std::string fileType("msl");
    M_driver_static_bundle staticBundle{.mapped_data =
                                            M_driver_slice_from(vecAddMsl),
                                        .file_type = fileType.c_str(),
                                        .file_type_len = fileType.size()};
    M_driver_bundle_compilation_options opts{"", 0, -1};
    ASSERT_SUCCESS(bundle_load(context, &staticBundle, &opts, &library));
    std::string funcName("vec_add_msl");
    M_driver_function_t function;
    ASSERT_SUCCESS(function_load(context, library, funcName.c_str(),
                                 funcName.size(), &function));
    ASSERT_SUCCESS(function_unload(context, function));
  }
}

struct DriverMetalPluginFunctionTest : DriverPluginQueueTest {
  void SetUp() override {
    DriverPluginQueueTest::SetUp();
    if (this->HasFatalFailure() || this->IsSkipped())
      return;

    for (size_t i = 0; i < contexts.size(); i++) {
      M_driver_context_t context = contexts[i];
      M_driver_runtime_bundle_t library;
      std::string fileType("msl");
      M_driver_static_bundle staticBundle{.mapped_data =
                                              M_driver_slice_from(vecAddMsl),
                                          .file_type = fileType.c_str(),
                                          .file_type_len = fileType.size()};
      M_driver_bundle_compilation_options opts{"", 0, -1};
      ASSERT_SUCCESS(bundle_load(context, &staticBundle, &opts, &library));
      libraries.push_back(library);
      std::string funcName("vec_add_msl");
      M_driver_function_t function;
      ASSERT_SUCCESS(function_load(context, library, funcName.c_str(),
                                   funcName.size(), &function));
      functions.push_back(function);
    }
  }

  void TearDown() override {
    for (size_t i = 0; i < contexts.size(); i++) {
      M_driver_context_t context = contexts[i];
      M_driver_function_t function = functions[i];
      M_driver_runtime_bundle_t library = libraries[i];
      EXPECT_SUCCESS(function_unload(context, function));
      EXPECT_SUCCESS(bundle_unload(context, library));
    }
    DriverPluginQueueTest::TearDown();
  }

  std::vector<M_driver_runtime_bundle_t> libraries;
  std::vector<M_driver_function_t> functions;
};

TEST_F(DriverMetalPluginFunctionTest, QueueExecute) {
  for (size_t i = 0; i < contexts.size(); i++) {
    [[maybe_unused]] M_driver_context_t context = contexts[i];
    [[maybe_unused]] M_driver_queue_t queue = queues[i];
    [[maybe_unused]] M_driver_function_t function = functions[i];

    // Test the function with some data
    constexpr size_t length = 1024;
    constexpr size_t elemSize = sizeof(float);
    constexpr size_t memorySize = length * elemSize;

    // Cleanup allocated device memory
    std::vector<M_driver_memory_t> memories;
    auto defer = llvm::scope_exit([&]() {
      for (M_driver_memory_t memory : memories)
        memory_free_sync(context, memory);
    });

    // Create device buffers
    M_driver_memory_t in0Memory;
    ASSERT_SUCCESS(memory_alloc_sync(context, memorySize, &in0Memory));
    memories.push_back(in0Memory);

    M_driver_memory_t in1Memory;
    ASSERT_SUCCESS(memory_alloc_sync(context, memorySize, &in1Memory));
    memories.push_back(in1Memory);

    M_driver_memory_t outMemory;
    ASSERT_SUCCESS(memory_alloc_sync(context, memorySize, &outMemory));
    memories.push_back(outMemory);

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
    M_driver_memory_view toView0{in0Memory, /*offset=*/0, memorySize};
    ASSERT_SUCCESS(queue_copy_to_device(queue, &toView0, in0Host.data()));
    M_driver_memory_view toView1{in1Memory, /*offset=*/0, memorySize};
    ASSERT_SUCCESS(queue_copy_to_device(queue, &toView1, in1Host.data()));
    ASSERT_SUCCESS(queue_synchronize(queue));

    constexpr uint32_t blockDim = 256;
    M_driver_queue_execute_config_t config = {
        .mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU,
        .config = {.gpu = {
                       .grid = {length / blockDim, 1, 1},
                       .block = {blockDim, 1, 1},
                       .shared_mem_bytes = 0,
                       .attributes = nullptr,
                       .num_attributes = 0,
                   }}};

    void *outPtr = nullptr;
    ASSERT_SUCCESS(memory_property(outMemory, "address", &outPtr));
    ASSERT_NE(nullptr, outPtr);
    void *in0Ptr = nullptr;
    ASSERT_SUCCESS(memory_property(in0Memory, "address", &in0Ptr));
    ASSERT_NE(nullptr, in0Ptr);
    void *in1Ptr = nullptr;
    ASSERT_SUCCESS(memory_property(in1Memory, "address", &in1Ptr));
    ASSERT_NE(nullptr, in1Ptr);
    uint32_t lengthValue = static_cast<uint32_t>(length);
    void *args[] = {&outPtr, &in0Ptr, &in1Ptr, &lengthValue};
    uint64_t argSizes[] = {sizeof(outPtr), sizeof(in0Ptr), sizeof(in1Ptr),
                           sizeof(lengthValue)};

    ASSERT_SUCCESS(queue_execute(queue, function, &config, args, argSizes, 4));

    // Copy results back
    M_driver_memory_view fromView{outMemory, /*offset=*/0, memorySize};
    ASSERT_SUCCESS(queue_copy_from_device(queue, outHost.data(), &fromView));
    ASSERT_SUCCESS(queue_synchronize(queue));

    // Verify results
    for (size_t i = 0; i < length; ++i) {
      float expected = static_cast<float>(i) + 2.0f;
      EXPECT_FLOAT_EQ(outHost[i], expected)
          << "Mismatch at index " << i << ": expected " << expected << ", got "
          << outHost[i];
    }
  }
}

// Regression test: an invalid argument (null pointer or zero size) must be
// rejected with an error *before* a compute encoder is opened. The zero-size
// path previously returned mid-encode, leaving the encoder open; Metal asserts
// when a command buffer is dropped with an open encoder, aborting the process.
TEST_F(DriverMetalPluginFunctionTest, QueueExecuteRejectsInvalidArgs) {
  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_queue_t queue = queues[i];
    M_driver_function_t function = functions[i];

    M_driver_queue_execute_config_t config = {
        .mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU,
        .config = {.gpu = {
                       .grid = {1, 1, 1},
                       .block = {1, 1, 1},
                       .shared_mem_bytes = 0,
                       .attributes = nullptr,
                       .num_attributes = 0,
                   }}};

    // Four non-null argument pointers matching the kernel's argument count.
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;

    // A zero-sized argument is rejected — this is the path that used to leak an
    // open encoder.
    void *args[] = {&a0, &a1, &a2, &a3};
    uint64_t zeroSizeArgSizes[] = {sizeof(a0), 0, sizeof(a2), sizeof(a3)};
    EXPECT_NE(
        M_DRIVER_STATUS_SUCCESS,
        queue_execute(queue, function, &config, args, zeroSizeArgSizes, 4));

    // A null argument is rejected.
    void *nullArgs[] = {&a0, nullptr, &a2, &a3};
    uint64_t argSizes[] = {sizeof(a0), sizeof(a1), sizeof(a2), sizeof(a3)};
    EXPECT_NE(M_DRIVER_STATUS_SUCCESS,
              queue_execute(queue, function, &config, nullArgs, argSizes, 4));

    // The queue must still be usable afterwards: no encoder was leaked and no
    // command buffer was dropped with work pending.
    EXPECT_SUCCESS(queue_synchronize(queue));
  }
}

// Requesting more dynamic threadgroup (shared) memory than the device supports
// must be rejected. Prior to wiring up `shared_mem_bytes`, the field was
// silently ignored, so an absurd request would "succeed" and the kernel would
// run as if no shared memory had been asked for.
TEST_F(DriverMetalPluginFunctionTest, QueueExecuteOversizedSharedMemFails) {
  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];
    M_driver_function_t function = functions[i];

    constexpr size_t length = 1024;
    constexpr size_t memorySize = length * sizeof(float);

    std::vector<M_driver_memory_t> memories;
    auto defer = llvm::scope_exit([&]() {
      for (M_driver_memory_t memory : memories)
        memory_free_sync(context, memory);
    });

    M_driver_memory_t in0Memory, in1Memory, outMemory;
    ASSERT_SUCCESS(memory_alloc_sync(context, memorySize, &in0Memory));
    memories.push_back(in0Memory);
    ASSERT_SUCCESS(memory_alloc_sync(context, memorySize, &in1Memory));
    memories.push_back(in1Memory);
    ASSERT_SUCCESS(memory_alloc_sync(context, memorySize, &outMemory));
    memories.push_back(outMemory);

    void *outPtr = nullptr, *in0Ptr = nullptr, *in1Ptr = nullptr;
    ASSERT_SUCCESS(memory_property(outMemory, "address", &outPtr));
    ASSERT_SUCCESS(memory_property(in0Memory, "address", &in0Ptr));
    ASSERT_SUCCESS(memory_property(in1Memory, "address", &in1Ptr));
    uint32_t lengthValue = static_cast<uint32_t>(length);
    void *args[] = {&outPtr, &in0Ptr, &in1Ptr, &lengthValue};
    uint64_t argSizes[] = {sizeof(outPtr), sizeof(in0Ptr), sizeof(in1Ptr),
                           sizeof(lengthValue)};

    constexpr uint32_t blockDim = 256;
    // Far larger than any Apple GPU threadgroup memory limit (32 KiB today).
    constexpr uint32_t oversizedSharedMem = 1u << 30; // 1 GiB
    M_driver_queue_execute_config_t config = {
        .mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU,
        .config = {.gpu = {
                       .grid = {length / blockDim, 1, 1},
                       .block = {blockDim, 1, 1},
                       .shared_mem_bytes = oversizedSharedMem,
                       .attributes = nullptr,
                       .num_attributes = 0,
                   }}};

    M_driver_status_t status =
        queue_execute(queue, function, &config, args, argSizes, 4);
    EXPECT_NE(M_DRIVER_STATUS_SUCCESS, status)
        << "execute() must reject shared_mem_bytes that exceeds the device "
           "threadgroup memory limit";
  }
}

// A valid, in-limit, non-zero `shared_mem_bytes` request must still be accepted
// and produce correct results. This guards against the validation rejecting
// legitimate requests and exercises the setThreadgroupMemoryLength() binding
// path on a regular kernel.
TEST_F(DriverMetalPluginFunctionTest, QueueExecuteValidSharedMem) {
  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];
    M_driver_function_t function = functions[i];

    constexpr size_t length = 1024;
    constexpr size_t memorySize = length * sizeof(float);

    std::vector<M_driver_memory_t> memories;
    auto defer = llvm::scope_exit([&]() {
      for (M_driver_memory_t memory : memories)
        memory_free_sync(context, memory);
    });

    M_driver_memory_t in0Memory, in1Memory, outMemory;
    ASSERT_SUCCESS(memory_alloc_sync(context, memorySize, &in0Memory));
    memories.push_back(in0Memory);
    ASSERT_SUCCESS(memory_alloc_sync(context, memorySize, &in1Memory));
    memories.push_back(in1Memory);
    ASSERT_SUCCESS(memory_alloc_sync(context, memorySize, &outMemory));
    memories.push_back(outMemory);

    std::vector<float> in0Host(length), in1Host(length), outHost(length, 0.0f);
    for (size_t j = 0; j < length; ++j) {
      in0Host[j] = static_cast<float>(j);
      in1Host[j] = 2.0f;
    }

    M_driver_memory_view toView0{in0Memory, /*offset=*/0, memorySize};
    ASSERT_SUCCESS(queue_copy_to_device(queue, &toView0, in0Host.data()));
    M_driver_memory_view toView1{in1Memory, /*offset=*/0, memorySize};
    ASSERT_SUCCESS(queue_copy_to_device(queue, &toView1, in1Host.data()));
    ASSERT_SUCCESS(queue_synchronize(queue));

    void *outPtr = nullptr, *in0Ptr = nullptr, *in1Ptr = nullptr;
    ASSERT_SUCCESS(memory_property(outMemory, "address", &outPtr));
    ASSERT_SUCCESS(memory_property(in0Memory, "address", &in0Ptr));
    ASSERT_SUCCESS(memory_property(in1Memory, "address", &in1Ptr));
    uint32_t lengthValue = static_cast<uint32_t>(length);
    void *args[] = {&outPtr, &in0Ptr, &in1Ptr, &lengthValue};
    uint64_t argSizes[] = {sizeof(outPtr), sizeof(in0Ptr), sizeof(in1Ptr),
                           sizeof(lengthValue)};

    constexpr uint32_t blockDim = 256;
    // Modest, comfortably within the Apple GPU threadgroup memory limit.
    constexpr uint32_t validSharedMem = blockDim * sizeof(float); // 1 KiB
    M_driver_queue_execute_config_t config = {
        .mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU,
        .config = {.gpu = {
                       .grid = {length / blockDim, 1, 1},
                       .block = {blockDim, 1, 1},
                       .shared_mem_bytes = validSharedMem,
                       .attributes = nullptr,
                       .num_attributes = 0,
                   }}};

    ASSERT_SUCCESS(queue_execute(queue, function, &config, args, argSizes, 4));
    M_driver_memory_view fromView{outMemory, /*offset=*/0, memorySize};
    ASSERT_SUCCESS(queue_copy_from_device(queue, outHost.data(), &fromView));
    ASSERT_SUCCESS(queue_synchronize(queue));

    for (size_t j = 0; j < length; ++j) {
      float expected = static_cast<float>(j) + 2.0f;
      EXPECT_FLOAT_EQ(outHost[j], expected)
          << "Mismatch at index " << j << ": expected " << expected << ", got "
          << outHost[j];
    }
  }
}
