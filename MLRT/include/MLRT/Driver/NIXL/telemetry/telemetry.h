//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef MLRT_DRIVER_NIXL_TELEMETRY_TELEMETRY_H
#define MLRT_DRIVER_NIXL_TELEMETRY_TELEMETRY_H

#include "MLRT/Driver/NIXL/nixl_types.h"
#include "MLRT/Driver/NIXL/telemetry/telemetry_event.h"
#include "MLRT/Driver/NIXL/telemetry/telemetry_exporter.h"
#include "MLRT/Driver/NIXL/utils/common/cyclic_buffer.h"

#include <asio.hpp>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace M::AsyncRT::NIXL {

inline constexpr char TELEMETRY_ENABLED_VAR[] = "MODULAR_NIXL_TELEMETRY_ENABLE";
inline constexpr char TELEMETRY_DIR_VAR[] = "MODULAR_NIXL_TELEMETRY_DIR";

struct periodicTask {
  asio::steady_timer timer_;
  std::function<bool()> callback_;
  std::chrono::milliseconds interval_;
  std::atomic<bool> enabled_;

  periodicTask(const asio::any_io_executor &executor,
               std::chrono::milliseconds interval, bool enabled = false)
      : timer_(executor), callback_(nullptr), interval_(interval),
        enabled_(enabled) {}
};

class nixlTelemetry {
public:
  explicit nixlTelemetry(const std::string &agentName);
  nixlTelemetry(const nixlTelemetry &) = delete;
  nixlTelemetry(nixlTelemetry &&) = delete;
  nixlTelemetry &operator=(const nixlTelemetry &) = delete;
  nixlTelemetry &operator=(nixlTelemetry &&) = delete;
  ~nixlTelemetry();

  void updateTxBytes(uint64_t txBytes);
  void updateRxBytes(uint64_t rxBytes);
  void updateTxRequestsNum(uint32_t num);
  void updateRxRequestsNum(uint32_t num);
  void updateErrorCount(nixl_status_t errorType);
  void updateMemoryRegistered(uint64_t memoryRegistered);
  void updateMemoryDeregistered(uint64_t memoryDeregistered);
  void addXferTime(std::chrono::microseconds xferTime, bool isWrite,
                   uint64_t bytes);
  void addPostTime(std::chrono::microseconds postTime);
  void recordBackendEvents(std::vector<nixlTelemetryEvent> events);

private:
  void initializeTelemetry();
  void registerPeriodicTask(periodicTask &task);
  void enqueueEvent(const std::string &eventName,
                    nixl_telemetry_category_t category, uint64_t value);
  void enqueueEvent(const nixlTelemetryEvent &event);
  bool writeEventHelper();

  std::unique_ptr<nixlTelemetryExporter> exporter_;
  std::unique_ptr<sharedRingBuffer<nixlTelemetryEvent>> buffer_;
  std::vector<nixlTelemetryEvent> events_;
  std::mutex mutex_;
  asio::thread_pool pool_;
  periodicTask writeTask_;
  std::string agentName_;
  bool enabled_ = false;
};

} // namespace M::AsyncRT::NIXL

#endif // MLRT_DRIVER_NIXL_TELEMETRY_TELEMETRY_H
