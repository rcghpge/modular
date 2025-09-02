//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPPrimitives/KernelLogger.h"
#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>

namespace M {
namespace MGP {

/// Check if kernel logging is enabled via Runtime or MAX_LOG_KERNELS
/// environment variable
bool isKernelLoggingEnabled(AsyncRT::Runtime &runtime) {
  // Check the Runtime first if available
  if (runtime.isKernelLoggingEnabled())
    return true;

  // Fall back to environment variable
  static bool enabled = []() {
    if (const char *env = std::getenv("MAX_LOG_KERNELS")) {
      std::string value(env);
      return (value == "1" || value == "true" || value == "True" ||
              value == "yes" || value == "Yes");
    }
    return false;
  }();
  return enabled;
}

/// Generate unique kernel ID to correlate async launches and completions for
/// logging
uint64_t generateKernelId() {
  static std::atomic<uint64_t> counter{1};
  return counter.fetch_add(1, std::memory_order_relaxed);
}

/// Get relative timestamp in milliseconds since first kernel launch
double getRelativeTimestampMs() {
  static auto startTime = std::chrono::high_resolution_clock::now();
  auto now = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(now - startTime);
  return duration.count() / 1000.0; // Convert microseconds to milliseconds
}

/// Thread-safe kernel logging function to prevent interleaved output
void logKernelEvent(llvm::StringRef eventType, llvm::StringRef kernelName,
                    llvm::StringRef deviceInfo, uint64_t kernelId,
                    double timestampMs, llvm::raw_ostream &os) {
  static std::mutex logMutex;
  std::lock_guard<std::mutex> lock(logMutex);

  os << "[KERNEL] " << eventType << " " << kernelName << " on " << deviceInfo
     << " [id=" << kernelId << "] at +" << llvm::format("%.3f", timestampMs)
     << "ms\n";
}

/// Check if a kernel should be logged (includes enabled check and filtering)
bool shouldLogKernel(llvm::StringRef kernelName, AsyncRT::Runtime &runtime) {
  // Early return if kernel logging is disabled
  if (!isKernelLoggingEnabled(runtime))
    return false;

  // Filter out low-level primitives that don't correspond to graph operations
  static const llvm::DenseSet<llvm::StringRef> filteredKernels = {
      "index.constant",      "mgp.buffer.alloc",       "mgp.buffer.free",
      "mgp.buffer.constant", "mgp.tensor.from_buffer", "mgp.tensor.to_buffer"};

  return !filteredKernels.contains(kernelName);
}

/// Extract device contexts from operands
template <typename OperandContainer>
llvm::SmallVector<M::AsyncRT::DeviceContextRef>
extractDeviceContexts(const OperandContainer &operands) {
  llvm::SmallVector<M::AsyncRT::DeviceContextRef> deviceContexts;

  for (const M::AsyncRT::AnyAsyncValueRef &operand : operands) {
    if (auto deviceCtxRef =
            llvm::dyn_cast<M::AsyncRT::DeviceContextRef>(operand)) {
      deviceContexts.push_back(*deviceCtxRef);
    }
  }

  return deviceContexts;
}

/// Get device information string for logging (CPU vs GPU:id)
template <typename OperandContainer>
std::string getDeviceInfoString(const OperandContainer &operands) {
  llvm::SmallVector<std::string> deviceInfos;

  auto deviceContexts = extractDeviceContexts(operands);
  for (const auto &deviceCtxRef : deviceContexts) {
    const auto &device = deviceCtxRef->device();
    std::string deviceInfo;

    if (device.api() == AsyncRT::Device::cpuAPI)
      deviceInfo = "CPU";
    else
      // All non-CPU devices are considered GPU with device ID
      deviceInfo = "GPU:" + std::to_string(device.id());

    deviceInfos.push_back(std::move(deviceInfo));
  }

  if (deviceInfos.empty())
    return "CPU"; // Default assumption if no device contexts found
  if (deviceInfos.size() == 1)
    return deviceInfos[0];
  // Multiple devices - join with comma
  std::string result = "Multi(";
  llvm::raw_string_ostream os(result);
  llvm::interleaveComma(deviceInfos, os);
  result += ")";
  return result;
}

// Explicit template instantiations for common operand container types
template llvm::SmallVector<M::AsyncRT::DeviceContextRef>
extractDeviceContexts<llvm::ArrayRef<M::AsyncRT::AnyAsyncValueRef>>(
    const llvm::ArrayRef<M::AsyncRT::AnyAsyncValueRef> &operands);

template std::string
getDeviceInfoString<llvm::ArrayRef<M::AsyncRT::AnyAsyncValueRef>>(
    const llvm::ArrayRef<M::AsyncRT::AnyAsyncValueRef> &operands);

template llvm::SmallVector<M::AsyncRT::DeviceContextRef>
extractDeviceContexts<llvm::SmallVector<M::AsyncRT::AnyAsyncValueRef>>(
    const llvm::SmallVector<M::AsyncRT::AnyAsyncValueRef> &operands);

template std::string
getDeviceInfoString<llvm::SmallVector<M::AsyncRT::AnyAsyncValueRef>>(
    const llvm::SmallVector<M::AsyncRT::AnyAsyncValueRef> &operands);

template llvm::SmallVector<M::AsyncRT::DeviceContextRef>
extractDeviceContexts<M::RemainingArguments>(
    const M::RemainingArguments &operands);

template std::string getDeviceInfoString<M::RemainingArguments>(
    const M::RemainingArguments &operands);

/// KernelLogger implementation
template <typename OperandContainer, typename LoggingPolicy>
KernelLogger<OperandContainer, LoggingPolicy>::KernelLogger(
    llvm::StringRef kernelName, const OperandContainer &operands,
    llvm::ArrayRef<AsyncRT::AnyAsyncValueRef> outResults,
    AsyncRT::Runtime &runtime, llvm::raw_ostream &os)
    : outResults(outResults), outputStream(os),
      shouldLog(shouldLogKernel(kernelName, runtime)) {
  if (shouldLog) {
    this->kernelId = generateKernelId();
    this->kernelName = kernelName.str();
    this->deviceInfo = getDeviceInfoString(operands);
    double timestampMs = getRelativeTimestampMs();
    logKernelEvent(LoggingPolicy::launchEvent(), this->kernelName,
                   this->deviceInfo, this->kernelId, timestampMs, outputStream);
  }
}

template <typename OperandContainer, typename LoggingPolicy>
KernelLogger<OperandContainer, LoggingPolicy>::~KernelLogger() {
  if (shouldLog) {
    double timestampMs = getRelativeTimestampMs();
    if (llvm::any_of(outResults, [](const AsyncRT::AnyAsyncValueRef &result) {
          return result.isError();
        })) {
      logKernelEvent(LoggingPolicy::errorEvent(), kernelName, deviceInfo,
                     kernelId, timestampMs, outputStream);
    } else {
      logKernelEvent(LoggingPolicy::completeEvent(), kernelName, deviceInfo,
                     kernelId, timestampMs, outputStream);
    }
  }
}

// Explicit template instantiations for common types
template class KernelLogger<llvm::ArrayRef<M::AsyncRT::AnyAsyncValueRef>,
                            SyncLoggingPolicy>;
template class KernelLogger<llvm::ArrayRef<M::AsyncRT::AnyAsyncValueRef>,
                            AsyncLoggingPolicy>;
template class KernelLogger<llvm::SmallVector<M::AsyncRT::AnyAsyncValueRef>,
                            SyncLoggingPolicy>;
template class KernelLogger<llvm::SmallVector<M::AsyncRT::AnyAsyncValueRef>,
                            AsyncLoggingPolicy>;
template class KernelLogger<M::RemainingArguments, SyncLoggingPolicy>;
template class KernelLogger<M::RemainingArguments, AsyncLoggingPolicy>;

} // namespace MGP
} // namespace M
