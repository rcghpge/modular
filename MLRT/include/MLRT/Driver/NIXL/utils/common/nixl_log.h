//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef MLRT_DRIVER_NIXL_UTILS_COMMON_NIXL_LOG_H
#define MLRT_DRIVER_NIXL_UTILS_COMMON_NIXL_LOG_H

#include "MLRT/Driver/NIXL/nixl_types.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <ios>
#include <ostream>
#include <system_error>
#include <type_traits>

namespace M::AsyncRT::NIXL {

enum class NixlLogLevel {
  kTrace = 0,   // Map to VLOG(2)
  kDebug = 1,   // Map to VLOG(1)
  kInfo = 2,    // Map to INFO, VLOG(0)
  kWarning = 3, // Map to WARNING
  kError = 4,   // Map to ERROR
  kFatal = 5    // Map to FATAL
};

bool shouldLog(NixlLogLevel level);

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
  NixlLogStream(llvm::raw_ostream &os) : os_(os) {}
  ~NixlLogStream() { os_ << '\n'; }

  NixlLogStream &operator<<(std::ios_base &(*manip)(std::ios_base &)) {
    if (manip == static_cast<std::ios_base &(*)(std::ios_base &)>(std::hex)) {
      hexMode_ = true;
    } else if (manip ==
               static_cast<std::ios_base &(*)(std::ios_base &)>(std::dec)) {
      hexMode_ = false;
    } else if (manip == static_cast<std::ios_base &(*)(std::ios_base &)>(
                            std::showbase)) {
      showBase_ = true;
    } else if (manip == static_cast<std::ios_base &(*)(std::ios_base &)>(
                            std::noshowbase)) {
      showBase_ = false;
    }
    return *this;
  }

  NixlLogStream &operator<<(std::ostream &(*manip)(std::ostream &)) {
    if (manip == static_cast<std::ostream &(*)(std::ostream &)>(std::endl)) {
      os_ << '\n';
    }
    return *this;
  }

  NixlLogStream &operator<<(nixl_status_t value) {
    os_ << NixlEnumStrings::statusStr(value);
    return *this;
  }

  template <typename T>
    requires(std::is_enum_v<T>)
  NixlLogStream &operator<<(T value) {
    using U = std::underlying_type_t<T>;
    return *this << static_cast<U>(value);
  }

  template <typename T>
    requires(!std::is_enum_v<T>)
  NixlLogStream &operator<<(const T &value) {
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool> &&
                  !std::is_same_v<T, char> && !std::is_same_v<T, signed char> &&
                  !std::is_same_v<T, unsigned char>) {
      if (hexMode_) {
        if (showBase_) {
          os_ << "0x";
        }
        os_ << llvm::utohexstr(static_cast<std::make_unsigned_t<T>>(value));
        return *this;
      }
    }
    os_ << value;
    return *this;
  }

private:
  llvm::raw_ostream &os_;
  bool hexMode_ = false;
  bool showBase_ = false;
};

class NixlLogVoidify {
public:
  void operator&(NixlLogStream &) const {}
};

} // namespace M::AsyncRT::NIXL

/*-----------------------------------------------------------------------------*
 * Logging Macros
 *-----------------------------------------------------------------------------*
 * Ordered by severity (highest to lowest)
 * Usage: NIXL_INFO << "Message part 1 " << variable << " message part 2";
 */

#define NIXL_LOG_HEADER_ "[" << __FILE__ << ":" << __LINE__ << "] "

#define NIXL_LOG_IF_ENABLED_(level)                                            \
  !::M::AsyncRT::NIXL::shouldLog(level)                                        \
      ? (void)0                                                                \
      : ::M::AsyncRT::NIXL::NixlLogVoidify() &                                 \
            ::M::AsyncRT::NIXL::NixlLogStream(llvm::errs())                    \
                << NIXL_LOG_HEADER_

/*
 * Logs a message and terminates the program unconditionally.
 * Maps to LOG(FATAL). Use for unrecoverable errors.
 */
#define NIXL_FATAL                                                             \
  ::M::AsyncRT::NIXL::NixlLogStream(llvm::errs()) << NIXL_LOG_HEADER_

/*
 * Like NIXL_FATAL, but also prints the errno message.
 */
#define NIXL_PFATAL                                                            \
  ::M::AsyncRT::NIXL::NixlLogStream(llvm::errs())                              \
      << NIXL_LOG_HEADER_ << ": " << nixl_strerror(errno)

/* Logs messages unconditionally (maps to ERROR level) */
#define NIXL_ERROR                                                             \
  NIXL_LOG_IF_ENABLED_(::M::AsyncRT::NIXL::NixlLogLevel::kError)

/*
 * Like NIXL_ERROR, but also prints the errno message.
 */
#define NIXL_PERROR                                                            \
  NIXL_LOG_IF_ENABLED_(::M::AsyncRT::NIXL::NixlLogLevel::kError)               \
      << ": " << nixl_strerror(errno)

/*
 * Logs messages unconditionally (maps to WARNING level)
 */
#define NIXL_WARN                                                              \
  NIXL_LOG_IF_ENABLED_(::M::AsyncRT::NIXL::NixlLogLevel::kWarning)

/*
 * Like NIXL_WARN, but also prints the errno message.
 */
#define NIXL_PWARN                                                             \
  NIXL_LOG_IF_ENABLED_(::M::AsyncRT::NIXL::NixlLogLevel::kWarning)             \
      << ": " << nixl_strerror(errno)

/*
 * Logs messages unconditionally (maps to INFO level)
 */
#define NIXL_INFO NIXL_LOG_IF_ENABLED_(::M::AsyncRT::NIXL::NixlLogLevel::kInfo)

/*
 * Like NIXL_INFO, but also prints the errno message.
 */
#define NIXL_PINFO                                                             \
  NIXL_LOG_IF_ENABLED_(::M::AsyncRT::NIXL::NixlLogLevel::kInfo)                \
      << ": " << nixl_strerror(errno)

/*
 * Logs messages unconditionally (maps to verbosity level 1)
 */
#define NIXL_DEBUG                                                             \
  NIXL_LOG_IF_ENABLED_(::M::AsyncRT::NIXL::NixlLogLevel::kDebug)

/*
 * Like NIXL_DEBUG, but also prints the errno message.
 */
#define NIXL_PDEBUG                                                            \
  NIXL_LOG_IF_ENABLED_(::M::AsyncRT::NIXL::NixlLogLevel::kDebug)               \
      << ": " << nixl_strerror(errno)

/*
 * Logs messages unconditionally (maps to verbosity level 2)
 * Stripped from release builds.
 */
#define NIXL_TRACE                                                             \
  NIXL_LOG_IF_ENABLED_(::M::AsyncRT::NIXL::NixlLogLevel::kTrace)

/*
 * Like NIXL_TRACE, but also prints the errno message.
 */
#define NIXL_PTRACE                                                            \
  NIXL_LOG_IF_ENABLED_(::M::AsyncRT::NIXL::NixlLogLevel::kTrace)               \
      << ": " << nixl_strerror(errno)

#endif // MLRT_DRIVER_NIXL_UTILS_COMMON_NIXL_LOG_H
