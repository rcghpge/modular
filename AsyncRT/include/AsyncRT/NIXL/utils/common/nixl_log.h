//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNCRT_NIXL_UTILS_COMMON_NIXL_LOG_H
#define ASYNCRT_NIXL_UTILS_COMMON_NIXL_LOG_H

#include "llvm/Support/raw_ostream.h"
#include <atomic>  // For std::atomic
#include <sstream> // For std::ostringstream

namespace M::AsyncRT::NIXL {

enum class NixlLogLevel {
  kTrace = 0,   // Map to VLOG(2)
  kDebug = 1,   // Map to VLOG(1)
  kInfo = 2,    // Map to INFO, VLOG(0)
  kWarning = 3, // Map to WARNING
  kError = 4,   // Map to ERROR
  kFatal = 5    // Map to FATAL
};

} // namespace M::AsyncRT::NIXL

/*-----------------------------------------------------------------------------*
 * Logging Macros
 *-----------------------------------------------------------------------------*
 * Ordered by severity (highest to lowest)
 * Usage: NIXL_INFO << "Message part 1 " << variable << " message part 2";
 */

/*
 * Logs a message and terminates the program unconditionally.
 * Maps to LOG(FATAL). Use for unrecoverable errors.
 */
#define NIXL_FATAL llvm::errs()

/* Logs messages unconditionally (maps to ERROR level) */
#define NIXL_ERROR llvm::errs()

/*
 * Logs messages unconditionally (maps to WARNING level)
 */
#define NIXL_WARN llvm::errs()

/*
 * Logs messages unconditionally (maps to INFO level)
 */
#define NIXL_INFO llvm::errs()

/*
 * Logs messages unconditionally (maps to verbosity level 1)
 */
#define NIXL_DEBUG llvm::errs()

/*
 * Logs messages unconditionally (maps to verbosity level 2)
 * Stripped from release buids.
 */
#define NIXL_TRACE llvm::errs()

#endif // ASYNCRT_NIXL_UTILS_COMMON_NIXL_LOG_H
