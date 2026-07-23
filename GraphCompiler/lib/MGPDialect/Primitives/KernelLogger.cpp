//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Primitives/KernelLogger.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"
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

/// Extract device contexts from operands
template <typename OperandContainer>
llvm::SmallVector<M::Driver::DeviceContextRef>
extractDeviceContexts(const OperandContainer &operands) {
  llvm::SmallVector<M::Driver::DeviceContextRef> deviceContexts;

  for (const M::AsyncRT::AnyAsyncValueRef &operand : operands) {
    if (auto deviceCtxRef =
            llvm::dyn_cast<M::Driver::DeviceContextRef>(operand)) {
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

    if (device.api() == M::Driver::Device::cpuAPI)
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
template llvm::SmallVector<M::Driver::DeviceContextRef>
extractDeviceContexts<llvm::ArrayRef<M::AsyncRT::AnyAsyncValueRef>>(
    const llvm::ArrayRef<M::AsyncRT::AnyAsyncValueRef> &operands);

template std::string
getDeviceInfoString<llvm::ArrayRef<M::AsyncRT::AnyAsyncValueRef>>(
    const llvm::ArrayRef<M::AsyncRT::AnyAsyncValueRef> &operands);

template llvm::SmallVector<M::Driver::DeviceContextRef>
extractDeviceContexts<llvm::SmallVector<M::AsyncRT::AnyAsyncValueRef>>(
    const llvm::SmallVector<M::AsyncRT::AnyAsyncValueRef> &operands);

template std::string
getDeviceInfoString<llvm::SmallVector<M::AsyncRT::AnyAsyncValueRef>>(
    const llvm::SmallVector<M::AsyncRT::AnyAsyncValueRef> &operands);

template llvm::SmallVector<M::Driver::DeviceContextRef>
extractDeviceContexts<M::RemainingArguments>(
    const M::RemainingArguments &operands);

template std::string getDeviceInfoString<M::RemainingArguments>(
    const M::RemainingArguments &operands);

/// KernelLogger implementation
template <typename OperandContainer, typename LoggingPolicy>
KernelLogger<OperandContainer, LoggingPolicy>::KernelLogger(
    llvm::StringRef kernelName, const OperandContainer &operands,
    llvm::ArrayRef<AsyncRT::AnyAsyncValueRef> outResults, llvm::raw_ostream &os)
    : outResults(outResults), outputStream(os) {
  this->kernelId = generateKernelId();
  this->kernelName = kernelName.str();
  this->deviceInfo = getDeviceInfoString(operands);
  double timestampMs = getRelativeTimestampMs();
  logKernelEvent(LoggingPolicy::launchEvent(), this->kernelName,
                 this->deviceInfo, this->kernelId, timestampMs, outputStream);
}

template <typename OperandContainer, typename LoggingPolicy>
KernelLogger<OperandContainer, LoggingPolicy>::~KernelLogger() {
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
