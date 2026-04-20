//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "MLRT/Driver/NIXL/utils/common/nixl_log.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <atomic>
#include <cstdlib>
#include <string>
#include <string_view>

using namespace M;
using namespace M::MLRT;

namespace {

// Structure to hold logging settings
struct LogLevelSettings {
  NixlLogLevel min_severity;
  int vlog_level;
};

std::atomic<NixlLogLevel> gMinNixlLogLevel{NixlLogLevel::kWarning};
std::atomic<int> gNixlVlogLevel{0};

} // anonymous namespace

// Default log level if nothing else is specified
constexpr std::string_view kDefaultLogLevel = "WARN";

// Function to initialize logging, run before main() via constructor attribute.
static void InitializeNixlLogging() __attribute__((constructor));

static void InitializeNixlLogging() {
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
  LogLevelSettings settings = kLogLevelMap.lookup(kDefaultLogLevel);
  bool invalidEnvVar = false;

  // Check environment variable, it has priority over compile-time default.
  const char *envLogLevel = std::getenv("MODULAR_NIXL_LOG_LEVEL");
  if (envLogLevel != nullptr) {
    std::string envLevelStrUpper = StringRef(envLogLevel).upper();
    if (auto it = kLogLevelMap.find(envLevelStrUpper);
        it != kLogLevelMap.end()) {
      settings = it->second;
    } else {
      // Fall back to kDefaultLogLevel if env var is invalid
      invalidEnvVar = true;
    }
  }

  // Store the determined levels globally
  gMinNixlLogLevel.store(settings.min_severity, std::memory_order_relaxed);
  gNixlVlogLevel.store(settings.vlog_level, std::memory_order_relaxed);

  if (invalidEnvVar) {
    llvm::errs() << "Invalid MODULAR_NIXL_LOG_LEVEL environment variable ('"
                 << envLogLevel
                 << "'), using default log level: " << kDefaultLogLevel
                 << "\\n";
  }
}

namespace M::MLRT {

bool shouldLog(NixlLogLevel level) {
  if (level == NixlLogLevel::kFatal) {
    return true;
  }

  const NixlLogLevel minSeverity =
      gMinNixlLogLevel.load(std::memory_order_relaxed);
  const int vlogLevel = gNixlVlogLevel.load(std::memory_order_relaxed);

  switch (level) {
  case NixlLogLevel::kTrace:
    return minSeverity <= NixlLogLevel::kInfo && vlogLevel >= 2;
  case NixlLogLevel::kDebug:
    return minSeverity <= NixlLogLevel::kInfo && vlogLevel >= 1;
  case NixlLogLevel::kInfo:
  case NixlLogLevel::kWarning:
  case NixlLogLevel::kError:
    return level >= minSeverity;
  case NixlLogLevel::kFatal:
    return true;
  }

  return true;
}

} // namespace M::MLRT
