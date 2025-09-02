//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPPrimitives/KernelLogger.h"
#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <sstream>

using namespace M;
using namespace M::MGP;

namespace {

class KernelLoggerTest : public ::testing::Test {
protected:
  std::unique_ptr<AsyncRT::Runtime> runtime;

  void SetUp() override {
    runtime =
        AsyncRT::createUniqueRuntime(AsyncRT::RuntimeOptions().forDebug());
  }
};

TEST_F(KernelLoggerTest, SyncLoggerBehavesCorrectly) {
  // Set up environment for logging
  setenv("MAX_LOG_KERNELS", "1", 1);

  // Create empty operands and results for basic testing
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> operands;
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> results;
  results.push_back(AsyncRT::AnyAsyncValueRef::createReady<int>(*runtime, 42));

  // Capture log output
  std::string logOutput;
  llvm::raw_string_ostream logStream(logOutput);

  // Test that logger correctly logs sync events
  {
    KernelSyncLogger<decltype(operands)> logger("test_kernel", operands,
                                                results, *runtime, logStream);
    // Logger destructor will trigger completion log
  }

  // Validate log output contains both launch and completion events
  EXPECT_THAT(logOutput, ::testing::HasSubstr(
                             "[KERNEL] SYNC LAUNCH test_kernel on CPU [id="));
  EXPECT_THAT(logOutput, ::testing::HasSubstr(
                             "[KERNEL] SYNC COMPLETE test_kernel on CPU [id="));
  EXPECT_THAT(logOutput, ::testing::HasSubstr("ms"));

  unsetenv("MAX_LOG_KERNELS");
}

TEST_F(KernelLoggerTest, AsyncLoggerBehavesCorrectly) {
  setenv("MAX_LOG_KERNELS", "1", 1);

  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> operands;
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> results;
  results.push_back(AsyncRT::AnyAsyncValueRef::createReady<int>(*runtime, 42));

  // Capture log output
  std::string logOutput;
  llvm::raw_string_ostream logStream(logOutput);

  // Test that async logger works correctly
  {
    KernelAsyncLogger<decltype(operands)> logger("async_kernel", operands,
                                                 results, *runtime, logStream);
    // Logger destructor will trigger completion log
  }

  // Validate log output contains both launch and completion events with ASYNC
  // labels
  EXPECT_THAT(logOutput, ::testing::HasSubstr(
                             "[KERNEL] ASYNC LAUNCH async_kernel on CPU [id="));
  EXPECT_THAT(
      logOutput,
      ::testing::HasSubstr("[KERNEL] ASYNC COMPLETE async_kernel on CPU [id="));

  unsetenv("MAX_LOG_KERNELS");
}

TEST_F(KernelLoggerTest, LogsNormalCompletionEvent) {
  setenv("MAX_LOG_KERNELS", "1", 1);

  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> operands;
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> results;
  results.push_back(AsyncRT::AnyAsyncValueRef::createReady<int>(*runtime, 42));

  // Capture log output
  std::string logOutput;
  llvm::raw_string_ostream logStream(logOutput);

  // Test normal completion logging
  {
    KernelSyncLogger<decltype(operands)> logger("normal_kernel", operands,
                                                results, *runtime, logStream);
    // Logger destructor will trigger completion log
  }

  // Validate log output contains launch and completion events
  EXPECT_THAT(logOutput, ::testing::HasSubstr(
                             "[KERNEL] SYNC LAUNCH normal_kernel on CPU [id="));
  EXPECT_THAT(
      logOutput,
      ::testing::HasSubstr("[KERNEL] SYNC COMPLETE normal_kernel on CPU [id="));
  EXPECT_THAT(logOutput, ::testing::Not(::testing::HasSubstr("SYNC ERROR")));

  unsetenv("MAX_LOG_KERNELS");
}

TEST_F(KernelLoggerTest, EnvironmentVariableEnablesLogging) {
  // Set environment variable explicitly for this test
  setenv("MAX_LOG_KERNELS", "1", 1);

  // Runtime disable setting doesn't override environment variable
  runtime->setKernelLogging(false);

  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> operands;
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> results;
  results.push_back(AsyncRT::AnyAsyncValueRef::createReady<int>(*runtime, 42));

  // Capture log output
  std::string logOutput;
  llvm::raw_string_ostream logStream(logOutput);

  // Environment variable should enable logging even when runtime logging is
  // disabled
  {
    KernelSyncLogger<decltype(operands)> logger("env_enabled_kernel", operands,
                                                results, *runtime, logStream);
  }

  // Validate that logging occurred despite runtime being disabled
  EXPECT_THAT(logOutput,
              ::testing::HasSubstr(
                  "[KERNEL] SYNC LAUNCH env_enabled_kernel on CPU [id="));
  EXPECT_THAT(logOutput,
              ::testing::HasSubstr(
                  "[KERNEL] SYNC COMPLETE env_enabled_kernel on CPU [id="));

  unsetenv("MAX_LOG_KERNELS");
}

TEST_F(KernelLoggerTest, FiltersLowLevelKernels) {
  setenv("MAX_LOG_KERNELS", "1", 1);

  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> operands;
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> results;
  results.push_back(AsyncRT::AnyAsyncValueRef::createReady<int>(*runtime, 42));

  // Test filtered kernel names
  std::vector<std::string> filteredKernels = {
      "index.constant",      "mgp.buffer.alloc",       "mgp.buffer.free",
      "mgp.buffer.constant", "mgp.tensor.from_buffer", "mgp.tensor.to_buffer"};

  for (const auto &kernelName : filteredKernels) {
    // Capture log output
    std::string logOutput;
    llvm::raw_string_ostream logStream(logOutput);

    {
      KernelSyncLogger<decltype(operands)> logger(kernelName, operands, results,
                                                  *runtime, logStream);
    }

    // Filtered kernels should not produce any log output
    EXPECT_TRUE(logOutput.empty())
        << "Kernel " << kernelName
        << " should be filtered but produced: " << logOutput;
  }

  unsetenv("MAX_LOG_KERNELS");
}

TEST_F(KernelLoggerTest, GeneratesUniqueKernelIds) {
  setenv("MAX_LOG_KERNELS", "1", 1);

  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> operands;
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> results1;
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> results2;
  results1.push_back(AsyncRT::AnyAsyncValueRef::createReady<int>(*runtime, 42));
  results2.push_back(AsyncRT::AnyAsyncValueRef::createReady<int>(*runtime, 43));

  // Capture log output
  std::string logOutput;
  llvm::raw_string_ostream logStream(logOutput);

  {
    KernelSyncLogger<decltype(operands)> logger1("kernel1", operands, results1,
                                                 *runtime, logStream);
  }

  {
    KernelSyncLogger<decltype(operands)> logger2("kernel2", operands, results2,
                                                 *runtime, logStream);
  }

  // Parse kernel IDs from LAUNCH events to verify they are unique
  std::string output = logOutput;
  size_t pos1 = output.find("SYNC LAUNCH kernel1");
  ASSERT_NE(pos1, std::string::npos);
  size_t idPos1 = output.find("[id=", pos1);
  ASSERT_NE(idPos1, std::string::npos);
  size_t idEnd1 = output.find("]", idPos1);
  ASSERT_NE(idEnd1, std::string::npos);
  std::string id1Str = output.substr(idPos1 + 4, idEnd1 - idPos1 - 4);

  size_t pos2 = output.find("SYNC LAUNCH kernel2");
  ASSERT_NE(pos2, std::string::npos);
  size_t idPos2 = output.find("[id=", pos2);
  ASSERT_NE(idPos2, std::string::npos);
  size_t idEnd2 = output.find("]", idPos2);
  ASSERT_NE(idEnd2, std::string::npos);
  std::string id2Str = output.substr(idPos2 + 4, idEnd2 - idPos2 - 4);

  // Kernel IDs should be different
  EXPECT_NE(id1Str, id2Str);

  unsetenv("MAX_LOG_KERNELS");
}

TEST_F(KernelLoggerTest, RuntimeEnableOverridesEnvDisable) {
  // Ensure environment is disabled
  unsetenv("MAX_LOG_KERNELS");

  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> operands;
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> results;
  results.push_back(AsyncRT::AnyAsyncValueRef::createReady<int>(*runtime, 42));

  // Capture log output
  std::string logOutput;
  llvm::raw_string_ostream logStream(logOutput);

  // Test runtime enable overrides environment disable
  runtime->setKernelLogging(true);
  {
    KernelSyncLogger<decltype(operands)> logger("runtime_override", operands,
                                                results, *runtime, logStream);
  }

  // Validate that logging occurred when runtime is enabled
  EXPECT_THAT(logOutput,
              ::testing::HasSubstr(
                  "[KERNEL] SYNC LAUNCH runtime_override on CPU [id="));
  EXPECT_THAT(logOutput,
              ::testing::HasSubstr(
                  "[KERNEL] SYNC COMPLETE runtime_override on CPU [id="));

  runtime->setKernelLogging(false);
}

TEST_F(KernelLoggerTest, LogsCorrectKernelAndDeviceInfo) {
  setenv("MAX_LOG_KERNELS", "1", 1);

  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> operands;
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> results;
  results.push_back(AsyncRT::AnyAsyncValueRef::createReady<int>(*runtime, 42));

  // Capture log output
  std::string logOutput;
  llvm::raw_string_ostream logStream(logOutput);

  {
    KernelSyncLogger<decltype(operands)> logger("my_kernel", operands, results,
                                                *runtime, logStream);
  }

  // Validate kernel name and device info appear in log output
  EXPECT_THAT(logOutput, ::testing::HasSubstr("my_kernel"));
  EXPECT_THAT(logOutput, ::testing::HasSubstr("on CPU"));
  EXPECT_THAT(logOutput, ::testing::HasSubstr("[id="));

  unsetenv("MAX_LOG_KERNELS");
}

TEST_F(KernelLoggerTest, RuntimeKernelLoggingOverridesEnv) {
  // Disable via environment but enable via runtime
  unsetenv("MAX_LOG_KERNELS");
  runtime->setKernelLogging(true);

  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> operands;
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> results;
  results.push_back(AsyncRT::AnyAsyncValueRef::createReady<int>(*runtime, 42));

  // Capture log output
  std::string logOutput;
  llvm::raw_string_ostream logStream(logOutput);

  {
    KernelSyncLogger<decltype(operands)> logger("runtime_enabled", operands,
                                                results, *runtime, logStream);
  }

  // Should log because runtime logging is enabled
  EXPECT_THAT(
      logOutput,
      ::testing::HasSubstr("[KERNEL] SYNC LAUNCH runtime_enabled on CPU [id="));
  EXPECT_THAT(logOutput,
              ::testing::HasSubstr(
                  "[KERNEL] SYNC COMPLETE runtime_enabled on CPU [id="));

  runtime->setKernelLogging(false);
}

} // namespace
