//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "MLRT/AsyncRT/NIXL/utils/common/nixl_log.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <atomic>
#include <cstdlib>
#include <string>
#include <string_view>

using namespace M;
using namespace M::AsyncRT::NIXL;

namespace {

// Structure to hold logging settings
struct LogLevelSettings {
  NixlLogLevel min_severity;
  int vlog_level;
};

} // anonymous namespace

// Default log level if nothing else is specified
constexpr std::string_view kDefaultLogLevel = "WARN";

// Function to initialize logging, run before main() via constructor attribute.
static void InitializeNixlLogging() __attribute__((constructor));

static void InitializeNixlLogging() {
  // Define and initialize the global log level variables declared in the header
  // Default to Warning level (3) and vlog level 0
  static std::atomic<NixlLogLevel> g_min_nixl_log_level{NixlLogLevel::kWarning};
  static std::atomic<int> g_nixl_vlog_level{0};

  // Map from log level string to settings using LLVM's StringMap
  llvm::StringMap<LogLevelSettings> kLogLevelMap{
      {"TRACE", {NixlLogLevel::kInfo, 2}},
      {"DEBUG", {NixlLogLevel::kInfo, 1}},
      {"INFO", {NixlLogLevel::kInfo, 0}},
      {"WARN", {NixlLogLevel::kWarning, 0}},
      {"ERROR", {NixlLogLevel::kError, 0}},
      {"FATAL", {NixlLogLevel::kFatal, 0}},
  };

  // This is the fallback log level, an option of last resort if nothing else is
  // specified.
  std::string_view level_to_use = kDefaultLogLevel;
  bool invalid_env_var = false;

  // Check environment variable, it has priority over compile-time default.
  const char *env_log_level = std::getenv("MODULAR_NIXL_LOG_LEVEL");
  if (env_log_level != nullptr) {
    std::string env_level_str_upper = StringRef(env_log_level).upper();
    if (kLogLevelMap.count(env_level_str_upper)) {
      level_to_use = env_level_str_upper;
    } else {
      // Fall back to kDefaultLogLevel if env var is invalid
      invalid_env_var = true;
    }
  }

  // Apply the settings by finding the level in the map
  auto it = kLogLevelMap.find(level_to_use);
  const LogLevelSettings &settings =
      (it != kLogLevelMap.end()) ? it->second
                                 : kLogLevelMap.lookup(kDefaultLogLevel);

  // Store the determined levels globally
  g_min_nixl_log_level.store(settings.min_severity, std::memory_order_relaxed);
  g_nixl_vlog_level.store(settings.vlog_level, std::memory_order_relaxed);

  if (invalid_env_var) {
    llvm::errs() << "Invalid MODULAR_NIXL_LOG_LEVEL environment variable ('"
                 << env_log_level
                 << "'), using default log level: " << kDefaultLogLevel
                 << "\\n";
  }
}
