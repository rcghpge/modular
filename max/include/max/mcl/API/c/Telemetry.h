//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_API_TELEMETRY_H
#define GENERICML_API_TELEMETRY_H

#include "GraphCompiler/Driver/Common/DriverInputFormat.h"
#include "GraphCompiler/Support/Telemetry.h"
#include "Support/SymbolExport.h"

#include <string_view>

namespace M::Telemetry {

// Logs the type of model being compiled.
void emitCompiledModelType(Telemetry::TelemetryContext &telemetryCtx,
                           FileFormat format);

MODULAR_CXX_EXPORT void
emitCompileInfo(std::string_view eventName,
                GML::MaxEngineTelemetryContext &telemetryCtx,
                std::optional<Error> error);

// Logs the pipeline name if call originates with MAX Graph.
void emitPipelineName(std::string_view eventName,
                      Telemetry::TelemetryContext &telemetryCtx,
                      std::string_view pipelineName);

} // namespace M::Telemetry

#endif // GENERICML_API_TELEMETRY_H
