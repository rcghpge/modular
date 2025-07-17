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

// Helper class that adds a newline when destroyed
class NixlLogStream {
public:
  NixlLogStream(llvm::raw_ostream &os) : os_(os) {}
  ~NixlLogStream() { os_ << '\n'; }

  template <typename T>
  NixlLogStream &operator<<(const T &value) {
    os_ << value;
    return *this;
  }

private:
  llvm::raw_ostream &os_;
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
#define NIXL_FATAL M::AsyncRT::NIXL::NixlLogStream(llvm::errs())

/* Logs messages unconditionally (maps to ERROR level) */
#define NIXL_ERROR M::AsyncRT::NIXL::NixlLogStream(llvm::errs())

/*
 * Logs messages unconditionally (maps to WARNING level)
 */
#define NIXL_WARN M::AsyncRT::NIXL::NixlLogStream(llvm::errs())

/*
 * Logs messages unconditionally (maps to INFO level)
 */
#define NIXL_INFO M::AsyncRT::NIXL::NixlLogStream(llvm::errs())

/*
 * Logs messages unconditionally (maps to verbosity level 1)
 */
#define NIXL_DEBUG M::AsyncRT::NIXL::NixlLogStream(llvm::errs())

/*
 * Logs messages unconditionally (maps to verbosity level 2)
 * Stripped from release buids.
 */
#define NIXL_TRACE M::AsyncRT::NIXL::NixlLogStream(llvm::errs())

#endif // ASYNCRT_NIXL_UTILS_COMMON_NIXL_LOG_H
