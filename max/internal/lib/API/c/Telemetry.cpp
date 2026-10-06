//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "max/API/c/Telemetry.h"

using namespace M;

static std::string getFrameworkAsString(M::FileFormat format) {
  if (format == FileFormat::kMODialect)
    return "MaxGraph";

  if (format == FileFormat::kUnknown)
    return "Unknown";

  return "Internal";
}

void M::Telemetry::emitCompiledModelType(
    Telemetry::TelemetryContext &telemetryCtx, FileFormat format) {
  std::string framework = getFrameworkAsString(format);
  auto counter = telemetryCtx.createUInt64Counter(
      "engine.load_model", M::Telemetry::Level::L1, {},
      "Model framework being loaded into the engine");

  counter.add(1, {{"framework", std::string(framework)}});
}

void M::Telemetry::emitCompileInfo(std::string_view eventName,
                                   GML::MaxEngineTelemetryContext &telemetryCtx,
                                   std::optional<Error> error) {
  if (error) {
    GML::emitErrorTelemetry<GML::MaxEngineErrorCode::CompilationError>(
        telemetryCtx, std::move(*error));
  } else {
    auto logger = telemetryCtx.getLogger("compiler");
    logger->emitL1Event(eventName, {{"max.engine.api.language",
                                     telemetryCtx.getAPILanguage()}});
  }
}

void M::Telemetry::emitPipelineName(std::string_view eventName,
                                    Telemetry::TelemetryContext &telemetryCtx,
                                    std::string_view pipelineName) {
  auto logger = telemetryCtx.getLogger("compiler");
  logger->emitL1Event(eventName, {{"max.pipeline.name", pipelineName}});
}
