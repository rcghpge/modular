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
#include <atomic> // For std::atomic
#include <cstring>
#include <sstream> // For std::ostringstream
#include <system_error>

namespace M::AsyncRT::NIXL {

enum class NixlLogLevel {
  kTrace = 0,   // Map to VLOG(2)
  kDebug = 1,   // Map to VLOG(1)
  kInfo = 2,    // Map to INFO, VLOG(0)
  kWarning = 3, // Map to WARNING
  kError = 4,   // Map to ERROR
  kFatal = 5    // Map to FATAL
};

/*-----------------------------------------------------------------------------*
 * Helper Functions
 *-----------------------------------------------------------------------------*/

/*
 * Get the error message for the given error number. Thread-safe.
 * @param err: The error number.
 * @return: The error message.
 */
inline std::string nixl_strerror(int err) {
  return std::error_code(err, std::generic_category()).message();
}

// Helper class that adds a newline when destroyed
class NixlLogStream {
public:
  NixlLogStream(llvm::raw_ostream &os)
      : os_(os), with_errno_(false), errno_val_(0) {}
  ~NixlLogStream() {
    if (with_errno_) {
      os_ << ": " << nixl_strerror(errno_val_);
    }
    os_ << '\n';
  }

  template <typename T>
  NixlLogStream &operator<<(const T &value) {
    os_ << value;
    return *this;
  }

  NixlLogStream &WithPerror() {
    with_errno_ = true;
    errno_val_ = errno;
    return *this;
  }

private:
  llvm::raw_ostream &os_;
  bool with_errno_;
  int errno_val_;
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

/*
 * Like NIXL_FATAL, but also prints the errno message.
 */
#define NIXL_PFATAL M::AsyncRT::NIXL::NixlLogStream(llvm::errs()).WithPerror()

/* Logs messages unconditionally (maps to ERROR level) */
#define NIXL_ERROR M::AsyncRT::NIXL::NixlLogStream(llvm::errs())

/*
 * Like NIXL_ERROR, but also prints the errno message.
 */
#define NIXL_PERROR M::AsyncRT::NIXL::NixlLogStream(llvm::errs()).WithPerror()

/*
 * Logs messages unconditionally (maps to WARNING level)
 */
#define NIXL_WARN M::AsyncRT::NIXL::NixlLogStream(llvm::errs())

/*
 * Like NIXL_WARN, but also prints the errno message.
 */
#define NIXL_PWARN M::AsyncRT::NIXL::NixlLogStream(llvm::errs()).WithPerror()

/*
 * Logs messages unconditionally (maps to INFO level)
 */
#define NIXL_INFO M::AsyncRT::NIXL::NixlLogStream(llvm::errs())

/*
 * Like NIXL_INFO, but also prints the errno message.
 */
#define NIXL_PINFO M::AsyncRT::NIXL::NixlLogStream(llvm::errs()).WithPerror()

/*
 * Logs messages unconditionally (maps to verbosity level 1)
 */
#define NIXL_DEBUG M::AsyncRT::NIXL::NixlLogStream(llvm::errs())

/*
 * Like NIXL_DEBUG, but also prints the errno message.
 */
#define NIXL_PDEBUG M::AsyncRT::NIXL::NixlLogStream(llvm::errs()).WithPerror()

/*
 * Logs messages unconditionally (maps to verbosity level 2)
 * Stripped from release buids.
 */
#define NIXL_TRACE M::AsyncRT::NIXL::NixlLogStream(llvm::errs())

/*
 * Like NIXL_TRACE, but also prints the errno message.
 */
#define NIXL_PTRACE M::AsyncRT::NIXL::NixlLogStream(llvm::errs()).WithPerror()

#endif // ASYNCRT_NIXL_UTILS_COMMON_NIXL_LOG_H
