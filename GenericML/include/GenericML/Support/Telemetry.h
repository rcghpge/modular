//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_TELEMETRY_H
#define GENERICML_SUPPORT_TELEMETRY_H

#include "Support/Context.h"
#include "Support/Error.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/STLExtras.h"
#include "Support/Telemetry/Telemetry.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/raw_ostream.h"

namespace M::GML {

enum class MaxEngineErrorCode : uint32_t {
  // # Error in user supplied data.
  UserInputError = 0,

  // These are specific cases of the above error that we are separately
  // tracking.
  FileNotFoundError = 1,
  UnknownFormatError = 2,

  // Error in deserializing the framework to appropriate graph
  // level representation
  TFImportError = 100,
  TorchImportError = 101,

  // Error in lowering imported model to internal representation.
  GraphLoweringError = 1000,
  MissingOpError = 1001,

  // General case for compilation.
  CompilationError = 10000
};

std::string stringifyErrorCode(MaxEngineErrorCode code);

/// This is a convenience wrapper over telemetry counter to specialize
/// a counter for an error code. This helps us to store in a context
/// object and reuse across frameworks and models.
class ErrorCodeCounter {
public:
  ErrorCodeCounter(Telemetry::TelemetryContext &ctx)
      : counter(makeCounter(ctx)) {}

  template <MaxEngineErrorCode errorCode>
  void increment() {
    std::string category = stringifyErrorCode(errorCode);
    counter.add(1, {{category, to_underlying(errorCode)}});
  }

private:
  Telemetry::Counter<uint64_t> makeCounter(Telemetry::TelemetryContext &ctx) {
    return ctx.createUInt64Counter(
        "max.engine.error_code", M::Telemetry::Level::L1, {},
        "Category of failure emitted during model compilation.");
  }

  Telemetry::Counter<uint64_t> counter;
};

/// A convenience wrapper over TelemetryContext for MaxEngine-specific
/// telemetry.
class MaxEngineTelemetryContext {
public:
  MaxEngineTelemetryContext(M::ContextRef ctx)
      : context(ctx.getPointer()),
        telemetryContext(context->get<Telemetry::TelemetryContext>()) {
    assert(telemetryContext && "expected to be set up already");
  }

  ErrorCodeCounter &getErrorCounter() const {
    return context->emplaceIfMissing<ErrorCodeCounter>(*telemetryContext);
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
  // MaxEngineTelemetryContext is stored inside M::ContextRef and the
  // pointer here is for the parent M::Context. So this is safe to hold
  // as raw pointer.
  M::Context *context;
  Telemetry::TelemetryContext *telemetryContext;
  std::string apiLanguage;
};

/// Increment the count for a particular error code. This function will
/// internally increment a session scoped counter and emit telemetry metrics for
/// the error code. This function is useful if we want to count the error
/// occurrence but no associated error logs to emit along with it.
template <MaxEngineErrorCode ErrorCode>
void incrementTelemetryErrorCount(MaxEngineTelemetryContext &ctx) {
  auto &counter = ctx.getErrorCounter();
  counter.increment<ErrorCode>();
  ctx.flush();
}

/// Increment error count and emit error logs into appropriate telemetry end
/// points. The error logs can be found under the event domain `compiler` with
/// event name `compile.failure`.
template <MaxEngineErrorCode ErrorCode>
void emitErrorTelemetry(MaxEngineTelemetryContext &ctx, Error error) {
  incrementTelemetryErrorCount<ErrorCode>(ctx);

  std::shared_ptr<Telemetry::Logs::Logger> logger = ctx.getLogger("compiler");
  llvm::StringMap<Telemetry::Logs::AttributeValue> attributes = {
      {"error", error.get()}};
  std::string errorCodeStr = stringifyErrorCode(ErrorCode);
  attributes.insert({"max.engine.error_code", errorCodeStr});
  attributes.insert({"max.engine.api.language", ctx.getAPILanguage()});
  logger->emitL1Event("compile.failure", attributes);
}

/// Scoped handler that emits MLIR diagnostics to telemetry.
template <MaxEngineErrorCode ErrorCode>
struct MaxEngineTelemetryScope {
  MaxEngineTelemetryScope(mlir::MLIRContext *ctx,
                          GML::MaxEngineTelemetryContext *telemetryContext)
      : diagHandler(ctx, [telemetryContext](const mlir::Diagnostic &diag) {
          if (telemetryContext) {
            std::string errorMessage;
            llvm::raw_string_ostream os(errorMessage);
            diag.print(os);
            GML::emitErrorTelemetry<ErrorCode>(*telemetryContext,
                                               Error(errorMessage));
          }
          // Return failure to allow the main handler to process the diagnostic.
          return failure();
        }) {}

  mlir::ScopedDiagnosticHandler diagHandler;
};

} // namespace M::GML

#endif // GENERICML_SUPPORT_TELEMETRY_H
