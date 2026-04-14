//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "MLRT/Driver/NIXL/telemetry/telemetry.h"

#include "MLRT/Driver/NIXL/plugin_manager.h"
#include "MLRT/Driver/NIXL/utils/common/nixl_log.h"
#include "MLRT/Driver/NIXL/utils/common/str_tools.h"
#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>

namespace M::MLRT::NIXL {
namespace {

constexpr std::chrono::milliseconds kDefaultTelemetryRunInterval(100);
constexpr size_t kDefaultTelemetryBufferSize = 4096;
constexpr const char *kDefaultTelemetryPlugin = "BUFFER";

[[noreturn]] void telemetryFatal(const char *message) {
  llvm::report_fatal_error(message);
}

bool envVarEnabled(const char *name) {
  const char *value = std::getenv(name);
  if (!value) {
    return false;
  }

  return isEnabledFlagValue(value);
}

size_t telemetryBufferSize() {
  const char *value = std::getenv(TELEMETRY_BUFFER_SIZE_VAR);
  if (!value || *value == '\0') {
    return kDefaultTelemetryBufferSize;
  }

  const auto bufferSize = std::stoull(value);
  if (bufferSize == 0) {
    telemetryFatal("Telemetry buffer size cannot be 0");
  }
  return bufferSize;
}

std::chrono::milliseconds telemetryRunInterval() {
  const char *value = std::getenv(TELEMETRY_RUN_INTERVAL_VAR);
  if (!value || *value == '\0') {
    return kDefaultTelemetryRunInterval;
  }

  return std::chrono::milliseconds(std::stoll(value));
}

std::optional<std::string> getExporterName() {
  const char *exporterName = std::getenv(telemetryExporterVar);
  if (exporterName && *exporterName != '\0') {
    return std::string(exporterName);
  }

  const char *telemetryDir = std::getenv(TELEMETRY_DIR_VAR);
  if (!telemetryDir || *telemetryDir == '\0') {
    NIXL_DEBUG << TELEMETRY_DIR_VAR
               << " is not set, NIXL telemetry is enabled without any exporter";
    return std::nullopt;
  }

  NIXL_INFO << "No telemetry exporter was specified, using default: "
            << kDefaultTelemetryPlugin;
  return std::string(kDefaultTelemetryPlugin);
}

uint64_t timestampUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

} // namespace

nixlTelemetry::nixlTelemetry(const std::string &agentName)
    : pool_(1),
      writeTask_(pool_.get_executor(), kDefaultTelemetryRunInterval, false),
      agentName_(agentName) {
  if (agentName_.empty()) {
    telemetryFatal("Expected non-empty agent name in nixl telemetry create");
  }
  initializeTelemetry();
}

nixlTelemetry::~nixlTelemetry() {
  writeTask_.enabled_ = false;
  asio::error_code ec;
  writeTask_.timer_.cancel(ec);
  if (ec) {
    NIXL_DEBUG << "Failed to cancel telemetry write timer: " << ec.message();
  }
  pool_.stop();
  pool_.join();

  if (exporter_) {
    writeEventHelper();
  }
}

void nixlTelemetry::initializeTelemetry() {
  if (!envVarEnabled(TELEMETRY_ENABLED_VAR)) {
    return;
  }

  const size_t bufferSize = telemetryBufferSize();
  const auto exporterName = getExporterName();
  if (!exporterName) {
    return;
  }

  const char *telemetryDir = std::getenv(TELEMETRY_DIR_VAR);
  if (telemetryDir && *telemetryDir != '\0') {
    std::filesystem::create_directories(std::filesystem::path(telemetryDir));
  }

  auto &pluginManager = nixlPluginManager::getInstance();
  auto pluginHandle = pluginManager.loadTelemetryPlugin(*exporterName);
  if (!pluginHandle) {
    telemetryFatal("Failed to load telemetry plugin");
  }

  nixlTelemetryExporterInitParams initParams{
      .agentName = agentName_,
      .maxEventsBuffered = bufferSize,
  };
  exporter_ = pluginHandle->createExporter(initParams);
  if (!exporter_) {
    telemetryFatal("Failed to create telemetry exporter");
  }

  NIXL_DEBUG << "NIXL telemetry is enabled with exporter: " << *exporterName;

  writeTask_.callback_ = [this]() { return writeEventHelper(); };
  writeTask_.interval_ = telemetryRunInterval();
  writeTask_.enabled_ = true;
  registerPeriodicTask(writeTask_);
  enabled_ = true;
}

bool nixlTelemetry::writeEventHelper() {
  std::vector<nixlTelemetryEvent> nextQueue;
  if (exporter_) {
    nextQueue.reserve(exporter_->getMaxEventsBuffered());
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    events_.swap(nextQueue);
  }

  for (auto &event : nextQueue) {
    exporter_->exportEvent(event);
  }

  return true;
}

void nixlTelemetry::registerPeriodicTask(periodicTask &task) {
  task.timer_.expires_after(task.interval_);
  task.timer_.async_wait([this, &task](const asio::error_code &ec) {
    if (ec != asio::error::operation_aborted) {
      task.callback_();

      if (!task.enabled_) {
        return;
      }

      registerPeriodicTask(task);
    }
  });
}

void nixlTelemetry::enqueueEvent(const std::string &eventName,
                                 nixl_telemetry_category_t category,
                                 uint64_t value) {
  const nixlTelemetryEvent event(timestampUs(), category, eventName, value);
  enqueueEvent(event);
}

void nixlTelemetry::enqueueEvent(const nixlTelemetryEvent &event) {
  if (!enabled_ || !exporter_) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  events_.push_back(event);
}

void nixlTelemetry::updateTxBytes(uint64_t txBytes) {
  enqueueEvent("agent_tx_bytes",
               nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER, txBytes);
}

void nixlTelemetry::updateRxBytes(uint64_t rxBytes) {
  enqueueEvent("agent_rx_bytes",
               nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER, rxBytes);
}

void nixlTelemetry::updateTxRequestsNum(uint32_t num) {
  enqueueEvent("agent_tx_requests_num",
               nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER, num);
}

void nixlTelemetry::updateRxRequestsNum(uint32_t num) {
  enqueueEvent("agent_rx_requests_num",
               nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER, num);
}

void nixlTelemetry::updateErrorCount(nixl_status_t errorType) {
  enqueueEvent(NixlEnumStrings::statusStr(errorType),
               nixl_telemetry_category_t::NIXL_TELEMETRY_ERROR, 1);
}

void nixlTelemetry::updateMemoryRegistered(uint64_t memoryRegistered) {
  enqueueEvent("agent_memory_registered",
               nixl_telemetry_category_t::NIXL_TELEMETRY_MEMORY,
               memoryRegistered);
}

void nixlTelemetry::updateMemoryDeregistered(uint64_t memoryDeregistered) {
  enqueueEvent("agent_memory_deregistered",
               nixl_telemetry_category_t::NIXL_TELEMETRY_MEMORY,
               memoryDeregistered);
}

void nixlTelemetry::addXferTime(std::chrono::microseconds xferTime,
                                bool isWrite, uint64_t bytes) {
  const uint64_t time = timestampUs();
  const char *bytesName = isWrite ? "agent_tx_bytes" : "agent_rx_bytes";
  const char *requestsName =
      isWrite ? "agent_tx_requests_num" : "agent_rx_requests_num";

  std::lock_guard<std::mutex> lock(mutex_);
  events_.emplace_back(time,
                       nixl_telemetry_category_t::NIXL_TELEMETRY_PERFORMANCE,
                       "agent_xfer_time", xferTime.count());
  events_.emplace_back(time, nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER,
                       bytesName, bytes);
  events_.emplace_back(time, nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER,
                       requestsName, 1);
}

void nixlTelemetry::addPostTime(std::chrono::microseconds postTime) {
  enqueueEvent("agent_xfer_post_time",
               nixl_telemetry_category_t::NIXL_TELEMETRY_PERFORMANCE,
               postTime.count());
}

void nixlTelemetry::recordBackendEvents(
    std::vector<nixlTelemetryEvent> events) {
  if (!enabled_ || !exporter_) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  events_.insert(events_.end(), std::make_move_iterator(events.begin()),
                 std::make_move_iterator(events.end()));
}

namespace NixlEnumStrings {

std::string telemetryCategoryStr(nixl_telemetry_category_t category) {
  static constexpr std::array<std::string_view, 8> kCategoryNames = {
      "NIXL_TELEMETRY_MEMORY",     "NIXL_TELEMETRY_TRANSFER",
      "NIXL_TELEMETRY_CONNECTION", "NIXL_TELEMETRY_BACKEND",
      "NIXL_TELEMETRY_ERROR",      "NIXL_TELEMETRY_PERFORMANCE",
      "NIXL_TELEMETRY_SYSTEM",     "NIXL_TELEMETRY_CUSTOM",
  };

  const auto index = static_cast<size_t>(category);
  if (index >= kCategoryNames.size()) {
    return "BAD_CATEGORY";
  }
  return std::string(kCategoryNames[index]);
}

} // namespace NixlEnumStrings
} // namespace M::MLRT::NIXL
