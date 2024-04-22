//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_TELEMETRY_H
#define GENERICML_SUPPORT_TELEMETRY_H

#include "Support/Context.h"
#include "Support/Error.h"
#include "Support/Telemetry/Telemetry.h"

namespace M::GML {

enum class MaxEngineErrorCode {
  // # Error in user supplied data.
  UserInputError,

  // These are specific cases of the above error that we are separately
  // tracking.
  FileNotFoundError,
  UnknownFormatError,

  // Error in deserializing the framework to appropriate graph
  // level representation
  TFImportError,
  TorchImportError,
  ONNXImportError,

  // Error in lowering imported model to internal representation.
  GraphLoweringError,
  MissingOpError,

  // General case for compilation.
  CompilationError
};

std::string stringifyErrorCode(MaxEngineErrorCode code);

/// This is a convenience wrapper over telemetry counter to specialize
/// a counter for an error code. This helps us to store in a context
/// object and reuse across frameworks and models.
template <MaxEngineErrorCode ErrorCode>
class ErrorCodeCounter {
public:
  ErrorCodeCounter(Telemetry::TelemetryContext &ctx)
      : counter(makeCounter(ctx, ErrorCode)) {}

  void increment() { counter.add(1); }

private:
  Telemetry::Counter<uint64_t> makeCounter(Telemetry::TelemetryContext &ctx,
                                           MaxEngineErrorCode errorCode) {
    std::string category = stringifyErrorCode(errorCode);
    return ctx.createUInt64Counter(
        category, M::Telemetry::Level::L1, {},
        "Category of failure emitted during model compilation.");
  }

  Telemetry::Counter<uint64_t> counter;
};

/// A convenience wrapper over TelemetryContext for MaxEngine specific
/// telemetry.
class MaxEngineTelemetryContext {
public:
  MaxEngineTelemetryContext(M::ContextRef ctx)
      : context(std::move(ctx)),
        telemetryContext(context->get<Telemetry::TelemetryContext>()) {
    assert(telemetryContext && "expected to be setup already");
  }

  template <MaxEngineErrorCode ErrorCode>
  ErrorCodeCounter<ErrorCode> &getErrorCounter() const {
    return context->emplaceIfMissing<ErrorCodeCounter<ErrorCode>>(
        *telemetryContext);
  }

  std::shared_ptr<Telemetry::Logs::Logger>
  getLogger(const std::string &eventName) const {
    return telemetryContext->getLogger(eventName);
  }

  const std::string &getAPILanguage() const { return apiLanguage; }

  void setAPILanguage(const std::string &newAPILanguage) {
    apiLanguage = newAPILanguage;
  }

  void flush() { telemetryContext->flush(); }

private:
  M::ContextRef context;
  Telemetry::TelemetryContext *telemetryContext;
  std::string apiLanguage;
};

/// Increment the count for a particular error code. This function will
/// internally increment a session scoped counter and emit telemetry metrics for
/// the error code. This function is useful if we want to count the error
/// occurence but no associated error logs to emit along with it.
template <MaxEngineErrorCode ErrorCode>
void incrementTelemetryErrorCount(MaxEngineTelemetryContext &ctx) {
#ifdef MODULAR_ENABLE_TELEMETRY
  auto &counter = ctx.getErrorCounter<ErrorCode>();
  counter.increment();
  ctx.flush();
#endif // MODULAR_ENABLE_TELEMETRY
}

/// Increment error count and emit error logs into appropriate telemetry end
/// points. The error logs can be found under the event domain `compiler` with
/// event name `compile.failure`.
template <MaxEngineErrorCode ErrorCode>
void emitErrorTelemetry(MaxEngineTelemetryContext &ctx, Error error) {
#ifdef MODULAR_ENABLE_TELEMETRY
  incrementTelemetryErrorCount<ErrorCode>(ctx);

  std::shared_ptr<Telemetry::Logs::Logger> logger = ctx.getLogger("compiler");
  llvm::StringMap<Telemetry::Logs::AttributeValue> attributes = {
      {"error", error.get()}};
  std::string errorCodeStr = stringifyErrorCode(ErrorCode);
  attributes.insert({"max.engine.error_code", errorCodeStr});
  logger->emitL1Event("compile.failure", attributes);
#endif // MODULAR_ENABLE_TELEMETRY
}
} // namespace M::GML

#endif // GENERICML_SUPPORT_TELEMETRY_H
