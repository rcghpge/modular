//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Common error handling macros for DeviceContext implementations.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_DEVICECONTEXT_ERRORMACROS_H
#define MLRT_DRIVER_DEVICECONTEXT_ERRORMACROS_H

#include "MLRT/Driver/DeviceContext/DeviceContext.h"

/// Return a formatted error with file/line/function context.
/// Usage: RAISE_ERROR("Something failed");
#define RAISE_ERROR(msg)                                                       \
  do {                                                                         \
    return M::Driver::DeviceContext::formatError(__FILE__, __LINE__,           \
                                                 __FUNCTION__, msg);           \
  } while (0)

/// Evaluate an ErrorOr expression. If error, return formatted error.
/// Otherwise, define a new variable with the extracted value.
/// Usage: DEFINE_OR_RAISE(CUevent, event, cudaDevice.eventCachePop());
#define DEFINE_OR_RAISE(type, var, expr)                                       \
  auto _err_or_##var = (expr);                                                 \
  if (_err_or_##var.isError()) {                                               \
    return M::Driver::DeviceContext::formatError(                              \
        __FILE__, __LINE__, __FUNCTION__, _err_or_##var.getError());           \
  }                                                                            \
  type var = _err_or_##var.takeValue()

/// Evaluate an ErrorOrSuccess expression. If error, return formatted error.
/// Usage: ERR_CHECK_ERR(ctx.enqueueWaitForEvent(*srcEvent));
#define ERR_CHECK_ERR(expr)                                                    \
  do {                                                                         \
    auto _err_result_ = (expr);                                                \
    if (_err_result_.isError()) {                                              \
      return M::Driver::DeviceContext::formatError(                            \
          __FILE__, __LINE__, __FUNCTION__, _err_result_.getError());          \
    }                                                                          \
  } while (0)

/// Check for pending error on a DeviceContext.
/// Usage: CHECK_PENDING_ERR(*this);
#define CHECK_PENDING_ERR(ctx)                                                 \
  do {                                                                         \
    M::ErrorOrSuccess _pending_err_ = (ctx).checkPendingError();               \
    if (_pending_err_.isError())                                               \
      return _pending_err_.takeError();                                        \
  } while (0)

/// Assign to an existing variable from an ErrorOr expression.
/// Usage: ASSIGN_OR_RAISE(existingVar, someCall());
#define ASSIGN_OR_RAISE(var, expr)                                             \
  do {                                                                         \
    auto _err_or_var_ = (expr);                                                \
    if (_err_or_var_.isError()) {                                              \
      return M::Driver::DeviceContext::formatError(                            \
          __FILE__, __LINE__, __FUNCTION__, _err_or_var_.getError());          \
    }                                                                          \
    (var) = _err_or_var_.takeValue();                                          \
  } while (0)

/// Record pending error on context (used in async error handling).
/// This is for situations where we cannot return an error and must store it
/// for later reporting.
/// Usage: RECORD_PENDING_ERROR(ctx, someErrorOrCall());
#define RECORD_PENDING_ERROR(ctx, err)                                         \
  do {                                                                         \
    auto _record_err_result_ = (err);                                          \
    if (_record_err_result_.isError()) {                                       \
      M::Error _formatted_err_ = M::Driver::DeviceContext::formatError(        \
          __FILE__, __LINE__, __FUNCTION__, _record_err_result_.getError());   \
      (ctx).setPendingError(std::move(_formatted_err_));                       \
    }                                                                          \
  } while (0)

#endif // MLRT_DRIVER_DEVICECONTEXT_ERRORMACROS_H
